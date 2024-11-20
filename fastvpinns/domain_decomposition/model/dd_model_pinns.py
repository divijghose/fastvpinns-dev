'''
Domain Decomposition Model file for PINNs
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import initializers
import copy
import numpy as np

class PDDModel(tf.keras.Model):
    def __init__(self, layer_dims, learning_rate_dict, params_dict, loss_function, \
                 input_tensors_list, orig_factor_matrices, force_function_list, \
                 tensor_dtype, window_func_vals, x_limits, y_limits, activation = 'tanh',\
                 unnorm_freq = 2.0*np.pi):
        super(PDDModel, self).__init__()
        self.layer_dims = layer_dims
        self.learning_rate_dict = learning_rate_dict
        self.params_dict = params_dict
        self.loss_function = loss_function
        self.input_tensors_list = input_tensors_list
        self.orig_factor_matrices = orig_factor_matrices
        self.force_function_list = force_function_list
        self.tensor_dtype = tensor_dtype
        self.window_func_vals = window_func_vals
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.activation = activation

        self.layer_list = []
        self.x_min_limits = x_limits[0]
        self.x_max_limits = x_limits[1]
        self.y_min_limits = y_limits[0]
        self.y_max_limits = y_limits[1]

        self.mean_x = (self.x_min_limits + self.x_max_limits) / 2.0
        self.mean_y = (self.y_min_limits + self.y_max_limits) / 2.0
        self.semi_span_x = (self.x_max_limits - self.x_min_limits) / 2.0
        self.semi_span_y = (self.y_max_limits - self.y_min_limits) / 2.0

                # if dtype is not a valid tensorflow dtype, raise an error
        if not isinstance(self.tensor_dtype, tf.DType):
            raise TypeError("The given dtype is not a valid tensorflow dtype")

        self.pre_multiplier_val = copy.deepcopy(self.orig_factor_matrices[0])
        self.pre_multiplier_grad_x = copy.deepcopy(self.orig_factor_matrices[1])
        self.pre_multiplier_grad_y = copy.deepcopy(self.orig_factor_matrices[2])

        self.input_tensor = self.input_tensors_list[0]

        self.unnorm_freq = 2.0*np.pi

        # print(f"{'-'*74}")
        # print(f"| {'PARAMETER':<25} | {'SHAPE':<25} |")
        # print(f"{'-'*74}")
        # print(f"| {'input_tensor':<25} | {str(self.input_tensor.shape):<25} | {self.input_tensor.dtype}")
        # print(f"| {'force_matrix':<25} | {str(self.force_matrix.shape):<25} | {self.force_matrix.dtype}")
        # print(f"| {'pre_multiplier_grad_x':<25} | {str(self.pre_multiplier_grad_x.shape):<25} | {self.pre_multiplier_grad_x.dtype}")
        # print(f"| {'pre_multiplier_grad_y':<25} | {str(self.pre_multiplier_grad_y.shape):<25} | {self.pre_multiplier_grad_y.dtype}")
        # print(f"| {'pre_multiplier_val':<25} | {str(self.pre_multiplier_val.shape):<25} | {self.pre_multiplier_val.dtype}")
        # print(f"| {'dirichlet_input':<25} | {str(self.dirichlet_input.shape):<25} | {self.dirichlet_input.dtype}")
        # print(f"| {'dirichlet_actual':<25} | {str(self.dirichlet_actual.shape):<25} | {self.dirichlet_actual.dtype}")
        # print(f"{'-'*74}")

        self.n_cells = params_dict['n_cells']
        ## ----------------------------------------------------------------- ##
        ## ---------- LEARNING RATE AND OPTIMISER FOR THE MODEL ------------ ##
        ## ----------------------------------------------------------------- ##

        # parse the learning rate dictionary
        self.learning_rate_dict = learning_rate_dict
        initial_learning_rate = learning_rate_dict['initial_learning_rate']
        use_lr_scheduler = learning_rate_dict['use_lr_scheduler']
        decay_steps = learning_rate_dict['decay_steps']
        decay_rate = learning_rate_dict['decay_rate']
        staircase = learning_rate_dict['staircase']

        if(use_lr_scheduler):
            learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate, decay_steps, decay_rate, staircase=True
            )
        else:
            learning_rate_fn = initial_learning_rate

        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
        
        ## ----------------------------------------------------------------- ##
        ## --------------------- MODEL ARCHITECTURE ------------------------ ##
        ## ----------------------------------------------------------------- ##

        # Build dense layers based on the input list
        for dim in range(len(self.layer_dims) - 2):
            self.layer_list.append(layers.Dense(self.layer_dims[dim+1], activation=self.activation, \
                                                    kernel_initializer='glorot_uniform', \
                                                    dtype=self.tensor_dtype, bias_initializer='zeros'))
        
        # Add a output layer with no activation
        self.layer_list.append(layers.Dense(self.layer_dims[-1], activation = None, 
                                    kernel_initializer='glorot_uniform',
                                    dtype=self.tensor_dtype, bias_initializer='zeros'))
        

        # Add attention layer if required
        # if self.use_attention:
        #     self.attention_layer = layers.Attention()
        
        # Compile the model
        self.compile(optimizer=self.optimizer)
        self.build(input_shape=(None, self.layer_dims[0]))


    def call(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x

    def get_config(self):
        base_config = super().get_config()

        # Add the non-serializable arguments to the configuration
        base_config.update({
            'learning_rate_dict': self.learning_rate_dict,
            'loss_function': self.loss_function,
            'input_tensors_list':  self.input_tensors_list,
            'orig_factor_matrices': self.orig_factor_matrices,
            'force_function_list': self.force_function_list,
            'params_dict': self.params_dict,
            'use_attention': self.use_attention,
            'activation': self.activation,
            'hessian': self.hessian,
            'layer_dims': self.layer_dims,
            'tensor_dtype': self.tensor_dtype
        })

        return base_config
    
    @tf.function
    def pre_train_step(self, domain_decomposition, datahandler):
        
        with tf.GradientTape(persistent=True) as tape_outer:
            with tf.GradientTape(persistent=True) as tape_inner:

                tape_inner.watch(self.input_tensor)
                tape_outer.watch(self.input_tensor)

                x_values = self.input_tensor[:, 0:1]
                y_values = self.input_tensor[:, 1:2]

                normalized_x_values = (x_values - self.mean_x) / self.semi_span_x
                normalized_y_values = (y_values - self.mean_y) / self.semi_span_y

                normalized_input_tensor = tf.concat([normalized_x_values, normalized_y_values], axis=1)

                predicted_values = self(normalized_input_tensor)

                unnormalized_predicted_values = predicted_values * ((1/self.unnorm_freq)**2)

                unnnorm_pred_values_w_window_func = unnormalized_predicted_values * self.window_func_vals

                final_predicted_values = unnnorm_pred_values_w_window_func * tf.tanh(self.unnorm_freq*x_values) * \
                    tf.tanh(self.unnorm_freq*y_values) * tf.tanh(self.unnorm_freq*(1 - x_values)) * tf.tanh(self.unnorm_freq*(1 - y_values))
                
            gradients = tape_inner.gradient(final_predicted_values, self.input_tensor)
            pred_grad_x = gradients[:, 0:1]
            pred_grad_y = gradients[:, 1:2]
        
        pred_grad_2_x = tape_outer.gradient(pred_grad_x, self.input_tensor)
        pred_grad_2_y = tape_outer.gradient(pred_grad_y, self.input_tensor)

        pred_grad_x = tf.reshape(gradients[:, 0], [self.n_cells, self.pre_multiplier_grad_x.shape[-1]])
        pred_grad_y = tf.reshape(gradients[:, 1], [self.n_cells, self.pre_multiplier_grad_y.shape[-1]])
        pred_grad_xx = tf.reshape(pred_grad_2_x[:, 0], [self.n_cells, self.pre_multiplier_grad_x.shape[-1]])
        pred_grad_yy = tf.reshape(pred_grad_2_y[:, 1], [self.n_cells, self.pre_multiplier_grad_y.shape[-1]])
                                  

        pred_val = tf.reshape(predicted_values, [self.n_cells, self.pre_multiplier_val.shape[-1]])

       

        return pred_val, pred_grad_x, pred_grad_y, pred_grad_xx, pred_grad_yy

    
    @tf.function
    def train_step(self, beta=10, bilinear_params_dict=None, overlap_val=None, overlap_grad_x=None, \
                   overlap_grad_y=None, overlap_grad_grad_x=None, overlap_grad_grad_y=None):
        with tf.GradientTape(persistent=True) as training_tape:
            total_pde_loss = 0.0
            with tf.GradientTape(persistent=True) as tape_outer:
                with tf.GradientTape(persistent=True) as tape_inner:
                    tape_inner.watch(self.input_tensor)
                    tape_outer.watch(self.input_tensor)

                    x_values = self.input_tensor[:, 0:1]
                    y_values = self.input_tensor[:, 1:2]

                    normalized_x_values = (x_values - self.mean_x) / self.semi_span_x
                    normalized_y_values = (y_values - self.mean_y) / self.semi_span_y

                    normalized_input_tensor = tf.concat([normalized_x_values, normalized_y_values], axis=1)

                    predicted_values = self(normalized_input_tensor)

                    unnormalized_predicted_values = predicted_values * ((1/self.unnorm_freq)**2)

                    unnnorm_pred_values_w_window_func = unnormalized_predicted_values * self.window_func_vals

                    final_predicted_values = unnnorm_pred_values_w_window_func * tf.tanh(self.unnorm_freq*x_values) * \
                        tf.tanh(self.unnorm_freq*y_values) * tf.tanh(self.unnorm_freq*(1 - x_values)) * tf.tanh(self.unnorm_freq*(1 - y_values))
                
                gradients = tape_inner.gradient(final_predicted_values, self.input_tensor)
                pred_grad_x = gradients[:, 0:1]
                pred_grad_y = gradients[:, 1:2]
            
            pred_grad_2_x = tape_outer.gradient(pred_grad_x, self.input_tensor)
            pred_grad_2_y = tape_outer.gradient(pred_grad_y, self.input_tensor)

            pred_val = tf.reshape(final_predicted_values, [self.n_cells, self.pre_multiplier_val.shape[-1]])
            pred_grad_x = tf.reshape(pred_grad_x, [self.n_cells, self.pre_multiplier_grad_x.shape[-1]])
            pred_grad_y = tf.reshape(pred_grad_y, [self.n_cells, self.pre_multiplier_grad_y.shape[-1]])
            pred_grad_xx = tf.reshape(pred_grad_2_x[:, 0], [self.n_cells, self.pre_multiplier_grad_x.shape[-1]])
            pred_grad_yy = tf.reshape(pred_grad_2_y[:, 1], [self.n_cells, self.pre_multiplier_grad_y.shape[-1]])

            pred_val = pred_val + tf.stop_gradient(overlap_val)
            pred_grad_x = pred_grad_x + tf.stop_gradient(overlap_grad_x)
            pred_grad_y = pred_grad_y + tf.stop_gradient(overlap_grad_y)
            pred_grad_xx = pred_grad_xx + tf.stop_gradient(overlap_grad_grad_x)
            pred_grad_yy = pred_grad_yy + tf.stop_gradient(overlap_grad_grad_y)

            pred_val = tf.reshape(pred_val, [-1, 1])
            pred_grad_x = tf.reshape(pred_grad_x, [-1, 1])
            pred_grad_y = tf.reshape(pred_grad_y, [-1, 1])
            pred_grad_xx = tf.reshape(pred_grad_xx, [-1, 1])
            pred_grad_yy = tf.reshape(pred_grad_yy, [-1, 1])


            forcing_function = -2.0 * (self.unnorm_freq**2) * (tf.sin(self.unnorm_freq * x_values) * tf.sin(self.unnorm_freq * y_values))



            cells_residual = self.loss_function(forcing_function, pred_val, pred_grad_x, pred_grad_y, pred_grad_xx, pred_grad_yy)
            # total_residual = tf.reduce_mean(tf.square(cells_residual))
            total_residual = tf.reduce_sum(cells_residual)

            total_pde_loss = total_pde_loss + total_residual
        
        trainable_vars = self.trainable_variables
        self.gradients = training_tape.gradient(total_pde_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(self.gradients, trainable_vars))

        return total_pde_loss





    