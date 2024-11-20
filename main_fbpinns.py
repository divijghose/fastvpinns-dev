import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import yaml
import sys
import copy
from tensorflow.keras import layers
from tensorflow.keras import initializers
from rich.console import Console
import copy
import time
import optuna
import argparse


from fastvpinns.Geometry.geometry_2d import Geometry_2D
from fastvpinns.FE.fespace2d import Fespace2D
from fastvpinns.data.datahandler2d import DataHandler2D

from fastvpinns.domain_decomposition.model.dd_model_pinns import PDDModel
from fastvpinns.domain_decomposition.geometry.dd_geometry_2d import DDGeometry_2D
from fastvpinns.domain_decomposition.examples.sine_low_freq import *
from fastvpinns.domain_decomposition.domain_decomposition import *
from fastvpinns.domain_decomposition.scheduling import *
from fastvpinns.domain_decomposition.window_functions.window_functions import WindowFunction
from fastvpinns.domain_decomposition.fespace2d.decomposed_fespace2d import DecomposedFespace2D
from fastvpinns.domain_decomposition.data.datahandler_dd import DataHandler2D_DD
from fastvpinns.domain_decomposition.physics.poisson2d import pde_loss_poisson




# from fastvpinns.physics.poisson2d import pde_loss_poisson
from fastvpinns.utils.plot_utils import plot_contour, plot_loss_function, plot_test_loss_function
from fastvpinns.utils.compute_utils import compute_errors_combined
from fastvpinns.utils.print_utils import print_table, print_table_multicolumns


# import all files from utility
from utility import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FastVPINNs with YAML config or optimized hyperparameters"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="input.yaml",
        help="Path to YAML config file (default: input.yaml)",
    )
    parser.add_argument("--optimized", action="store_true", help="Use optimized hyperparameters")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=5000,
        help="Number of epochs to train each model in the hyperparameter optimization",
    )
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')

    if args.optimized:
        from fastvpinns.hyperparameter_tuning.optuna_tuner import OptunaTuner

        print("Running with optimized hyperparameters")
        print("This may take a while...")
        print("Running OptunaTuner...")

        tuner = OptunaTuner(n_trials=args.n_trials, n_jobs=len(gpus), n_epochs=args.n_epochs)
        best_params = tuner.run()
        # Convert best_params to the format expected by your code
        # config = convert_best_params_to_config(best_params)
        print("OptunaTuner completed")
        print("Best hyperparameters:")
        for key, value in best_params.items():
            print(f"{key}: {value}")
        sys.exit(0)
    elif args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("Please provide either a config file or use --optimized flag")
        sys.exit(1)

    console = Console()


    # # check input arguments
    # if len(sys.argv) != 2:
    #     print("Usage: python main.py <input file>")
    #     sys.exit(1)

    # # Read the YAML file
    # with open(sys.argv[1], 'r') as f:
    #     config = yaml.safe_load(f)

    # Extract the values from the YAML file
    # Values that are not hyperparameters:
    i_output_path = config['experimentation']['output_path']

    i_mesh_generation_method = config['geometry']['mesh_generation_method']
    if i_mesh_generation_method not in ["internal", "external", "convolution"]:
        raise ValueError("The given mesh generation method is not valid")
    
    i_generate_mesh_plot = config['geometry']['generate_mesh_plot']
    i_mesh_type = config['geometry']['mesh_type']
    i_x_min = config['geometry']['internal_mesh_params']['x_min']
    i_x_max = config['geometry']['internal_mesh_params']['x_max']
    i_y_min = config['geometry']['internal_mesh_params']['y_min']
    i_y_max = config['geometry']['internal_mesh_params']['y_max']
    i_n_cells_x = config['geometry']['internal_mesh_params']['n_cells_x']
    i_n_cells_y = config['geometry']['internal_mesh_params']['n_cells_y']

    if i_mesh_generation_method == "convolution":
        kernel_size_x = config['geometry']['convolution_mesh_params']['kernel_size_x']
        kernel_size_y = config['geometry']['convolution_mesh_params']['kernel_size_y']
        stride_x = config['geometry']['convolution_mesh_params']['stride_x']
        stride_y = config['geometry']['convolution_mesh_params']['stride_y']
        if(kernel_size_x > i_n_cells_x):
            raise ValueError("[ERROR] : kernel_size_x is greater than n_cells_x")
        if(kernel_size_y > i_n_cells_y):
            raise ValueError("[ERROR] : kernel_size_y is greater than n_cells_y")
        
    i_n_boundary_points = config['geometry']['internal_mesh_params']['n_boundary_points']
    i_n_test_points_x = config['geometry']['internal_mesh_params']['n_test_points_x']
    i_n_test_points_y = config['geometry']['internal_mesh_params']['n_test_points_y']

    i_mesh_file_name = config['geometry']['external_mesh_params']['mesh_file_name']
    i_boundary_refinement_level = config['geometry']['external_mesh_params']['boundary_refinement_level']
    i_boundary_sampling_method = config['geometry']['external_mesh_params']['boundary_sampling_method']

    i_fe_order = config['fe']['fe_order']
    i_fe_type = config['fe']['fe_type']
    i_quad_order = config['fe']['quad_order']
    i_quad_type = config['fe']['quad_type']

    i_model_architecture = config['model']['model_architecture']
    i_activation = config['model']['activation']
    i_use_attention = config['model']['use_attention']
    i_epochs = config['model']['epochs']
    i_dtype = config['model']['dtype']
    if i_dtype == "float64":
        i_dtype = tf.float64
    elif i_dtype == "float32":
        i_dtype = tf.float32
    else:
        print("[ERROR] The given dtype is not a valid tensorflow dtype")
        raise ValueError("The given dtype is not a valid tensorflow dtype")
    
    i_set_memory_growth = config['model']['set_memory_growth']
    i_learning_rate_dict = config['model']['learning_rate']

    i_beta = config['pde']['beta']

    i_update_progress_bar = config['logging']['update_progress_bar']
    i_update_console_output = config['logging']['update_console_output']
    i_update_solution_images = config['logging']['update_solution_images']

    i_use_wandb = config['logging']['use_wandb']
    i_wandb_project_name = config['logging']['project_name']
    i_wandb_run_prefix = config['logging']['wandb_run_prefix']
    i_wandb_entity = config['logging']['entity']
    
    print_verbose = config['logging']['print_verbose']

    i_scheduling_type = config['domain_decomposition']['scheduling_type']
    i_scheduling_freq = config['domain_decomposition']['scheduling_freq']
    i_window_function_scaling = config['domain_decomposition']['window_function_scaling']


    cell_dim_x = 1/i_n_cells_x
    cell_dim_y = 1/i_n_cells_y
    num_cell_overlap_x = kernel_size_x - stride_x
    num_cell_overlap_y = kernel_size_y - stride_y
    overlap_dim_x = num_cell_overlap_x * cell_dim_x
    overlap_dim_y = num_cell_overlap_y * cell_dim_y
    # use pathlib to create the folder,if it does not exist
    folder = Path(i_output_path)
    # create the folder if it does not exist
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    # get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()

    # Initiate a Geometry_2D object
    domain = DDGeometry_2D(
        i_mesh_type, i_mesh_generation_method, i_n_test_points_x, i_n_test_points_y, i_output_path
    )

    if i_mesh_generation_method == "convolution":

        cells_in_domain, blocks_in_domain, grid_x, grid_y = domain.generate_conv_domains(\
            x_limits = [i_x_min, i_x_max], y_limits = [i_y_min, i_y_max], \
            n_cells_x = i_n_cells_x, n_cells_y = i_n_cells_y, \
            kernel_size_row = kernel_size_x, kernel_size_col = kernel_size_y, \
            stride_row = stride_x, stride_col = stride_y)
        
        cells_points, boundary_limits = domain.generate_quad_mesh_convolutional(x_limits = [i_x_min, i_x_max], \
                                                            y_limits = [i_y_min, i_y_max], \
                                                            n_cells_x =  i_n_cells_x, \
                                                            n_cells_y = i_n_cells_y, \
                                                            num_boundary_points=i_n_boundary_points)
    else:
        raise ValueError("The given mesh generation method is not valid for domain decomposition")


    
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()

    # get the boundary function dictionary from example file

    cells_points_subdomain = {}
    for i in range(len(blocks_in_domain)):
        cells_points_subdomain[i] = domain.assign_sub_domain_coords(blocks_in_domain[i], cells_points)


    x_mean_list, y_mean_list = domain.calculate_subdomain_means()
    x_span_list, y_span_list = domain.calculate_subdomain_spans()


    decomposed_domain = DomainDecomposition(domain)

    scheduling_type = i_scheduling_type
    scheduling = Scheduling(domain, decomposed_domain, scheduling_type)
    scheduler_list = scheduling.scheduler_list

    window_function = WindowFunction(x_mean_list, x_span_list, y_mean_list, y_span_list, scaling_factor=i_window_function_scaling, overlap_dim=overlap_dim_x)
    x_min_list, x_max_list = window_function.x_min, window_function.x_max
    y_min_list, y_max_list = window_function.y_min, window_function.y_max

    #concatenate x_min_list and x_max_list to make two columns
    x_min_max_list = np.c_[x_min_list, x_max_list]
    y_min_max_list = np.c_[y_min_list, y_max_list]

    

    # get the boundary function dictionary from example file
    bound_function_dict, bound_condition_dict = get_boundary_function_dict(), get_bound_cond_dict()

    for i in range(len(blocks_in_domain)):
        window_function.plot_window_function(i, output_path=i_output_path)

    
        # generate dict for fespaces
    fespace = {}
    for i in range(len(blocks_in_domain)):
        fespace[i] = DecomposedFespace2D(cells_points=cells_points_subdomain[i],  \
                        cell_type=domain.mesh_type, fe_order=i_fe_order, fe_type =i_fe_type ,quad_order=i_quad_order, quad_type = i_quad_type, \
                        fe_transformation_type="bilinear",
                        forcing_function=rhs, output_path=i_output_path, x_minmax = x_min_max_list[i], y_minmax=  y_min_max_list[i])


    # instantiate data handler
    datahandler = {}
    for i in range(len(blocks_in_domain)):
        datahandler[i] = DataHandler2D_DD(fespace[i], domain, i, dtype=i_dtype)

    
    val_bound_tensors = {}
    gradx_bound_tensors = {}
    grady_bound_tensors = {}
    for i in range(len(blocks_in_domain)):
        val_bound_tensors[i], gradx_bound_tensors[i], grady_bound_tensors[i] = datahandler[i].get_hard_boundary_constraints()

    
    

    # Initialise params
    params_dict = {}

    for i in range(len(blocks_in_domain)):
        params_dict[i] = {}
        params_dict[i]['n_cells'] = fespace[i].n_cells
    

    # obtain the boundary points for dirichlet boundary conditions
    train_dirichlet_input, train_dirichlet_output = {}, {}
    for i in range(len(blocks_in_domain)):
        train_dirichlet_input[i], train_dirichlet_output[i] = tf.zeros(datahandler[i].x_pde_list.shape), tf.zeros(datahandler[i].x_pde_list.shape)
    

    # obtain bilinear params dict
    bilinear_params_dict = {}
    for i in range(len(blocks_in_domain)):
        bilinear_params_dict[i] = datahandler[i].get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)

    window_function_dict = {}
    for i in range(len(blocks_in_domain)):
        window_function_dict[i] = datahandler[i].get_window_func_vals(window_function)


    models_subdomain = {}
    for i in range(len(blocks_in_domain)):
        models_subdomain[i] = PDDModel(layer_dims = i_model_architecture, learning_rate_dict = i_learning_rate_dict, \
                            params_dict = params_dict[i], \
                            loss_function = pde_loss_poisson, input_tensors_list = [datahandler[i].x_pde_list, val_bound_tensors[i], gradx_bound_tensors[i], grady_bound_tensors[i]], \
                                orig_factor_matrices = [datahandler[i].shape_val_mat_list , datahandler[i].grad_x_mat_list, datahandler[i].grad_y_mat_list], \
                                force_function_list=datahandler[i].forcing_function_list, \
                                tensor_dtype = i_dtype, window_func_vals=window_function_dict[i], x_limits=x_min_max_list[i], y_limits=y_min_max_list[i])

    test_points = domain.get_test_points()
    console.print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
    y_exact = exact_solution(test_points[:,0], test_points[:,1])

    def update_overlap_values_scheduled(domain_decomposition, datahandler, fespace, active_domains, fixed_domains, trained_domains):

        total_num_cells = len(domain_decomposition.blocks_shared_by_cells)
        num_quad = int(fespace[0].fe_cell[0].quad_xi.shape[0])
        
        global_overlap_values = tf.zeros((total_num_cells, num_quad), dtype=i_dtype)
        global_overlap_grad_x = tf.zeros((total_num_cells, num_quad), dtype=i_dtype)
        global_overlap_grad_y = tf.zeros((total_num_cells, num_quad), dtype=i_dtype)
        global_overlap_grad_xx = tf.zeros((total_num_cells, num_quad), dtype=i_dtype)
        global_overlap_grad_yy = tf.zeros((total_num_cells, num_quad), dtype=i_dtype)
      

        # loop over domains
        # for block_id in range(len(domain_decomposition.blocks_in_domain)):
        for block_id in trained_domains:

            # obtain the cells in the current subdomain as list
            cells_in_current_subdomain = domain_decomposition.blocks_in_domain[block_id]

            # convert and reshape the cells_in_current_subdomain to a tensor
            cells_in_current_subdomain = tf.reshape(tf.constant(cells_in_current_subdomain), [-1, 1])

            # obtain the value and grad matrix of the current subdomain
            shape_val_mat = datahandler[block_id].current_domain_values
            grad_x_mat = datahandler[block_id].current_domain_grad_x
            grad_y_mat = datahandler[block_id].current_domain_grad_y
            grad_xx_mat = datahandler[block_id].current_domain_grad_xx
            grad_yy_mat = datahandler[block_id].current_domain_grad_yy
  

            # now using the cells in the current subdomain as list of column indices, 
            # add the values of shape_val_mat, grad_x_mat and grad_y_mat to the global_overlap_dict
            global_overlap_values = tf.tensor_scatter_nd_add(global_overlap_values, indices = cells_in_current_subdomain, updates = shape_val_mat)
            global_overlap_grad_x = tf.tensor_scatter_nd_add(global_overlap_grad_x, indices = cells_in_current_subdomain, updates = grad_x_mat)
            global_overlap_grad_y = tf.tensor_scatter_nd_add(global_overlap_grad_y, indices = cells_in_current_subdomain, updates = grad_y_mat)
            global_overlap_grad_xx = tf.tensor_scatter_nd_add(global_overlap_grad_xx, indices = cells_in_current_subdomain, updates = grad_xx_mat)
            global_overlap_grad_yy = tf.tensor_scatter_nd_add(global_overlap_grad_yy, indices = cells_in_current_subdomain, updates = grad_yy_mat)      

        # once all updates are done, extract the necessary columns and subract it from current domain values
        # to obtain the overlap values
        for block_id in active_domains:
            # obtain the cells in the current subdomain as list
            cells_in_current_subdomain = domain_decomposition.blocks_in_domain[block_id]
            overlap_val = tf.gather(global_overlap_values, cells_in_current_subdomain)
            overlap_grad_x = tf.gather(global_overlap_grad_x, cells_in_current_subdomain)
            overlap_grad_y = tf.gather(global_overlap_grad_y, cells_in_current_subdomain)
            overlap_grad_xx = tf.gather(global_overlap_grad_xx, cells_in_current_subdomain)
            overlap_grad_yy = tf.gather(global_overlap_grad_yy, cells_in_current_subdomain)
         
            # subtract the values from the current domain values
            datahandler[block_id].reset_overlap_values()
            datahandler[block_id].current_domain_overlap_values = tf.subtract(overlap_val, datahandler[block_id].current_domain_values)
            datahandler[block_id].current_domain_overlap_grad_x = tf.subtract(overlap_grad_x, datahandler[block_id].current_domain_grad_x)
            datahandler[block_id].current_domain_overlap_grad_y = tf.subtract(overlap_grad_y, datahandler[block_id].current_domain_grad_y)
            datahandler[block_id].current_domain_overlap_grad_xx = tf.subtract(overlap_grad_xx, datahandler[block_id].current_domain_grad_xx)
            datahandler[block_id].current_domain_overlap_grad_yy = tf.subtract(overlap_grad_yy, datahandler[block_id].current_domain_grad_yy)

    
        pass

    def predict_and_compute_loss_scheduled( domain, blocks_in_domain, models_subdomain, test_points, y_exact, window_function, i_dtype, i_n_test_points_x, i_n_test_points_y, i_output_path, trained_domains, epoch, plot_subomain_solution: False):
        # obtain the predictions
        y_pred = tf.zeros((test_points.shape[0], 1), dtype=i_dtype)
        for i in trained_domains:
        # for i in range(len(blocks_in_domain)):
            # get the mean and std-dev of input to current subdomain
            x_mean, x_span = domain.x_mean_list[i], domain.x_span_list[i]
            y_mean, y_span = domain.y_mean_list[i], domain.y_span_list[i]

            # cast test points to i_dtype
            test_points = tf.cast(test_points, dtype=i_dtype)

            # normalise the inputs
            x_test = (test_points[:,0:1] - x_mean)/(x_span/2.0)
            y_test = (test_points[:,1:2] - y_mean)/(y_span/2.0)

            # x_test = test_points[:,0:1]
            # y_test = test_points[:,1:2]
            normalised_test_points = tf.concat([x_test, y_test], axis=1)

            # obtain the predictions
            y_pred_subdomain = models_subdomain[i].predict(normalised_test_points)

            # unnormalise the predictions
            y_pred_subdomain = (1.0/(2.0*np.pi)**2) * y_pred_subdomain

            # calculate the window functions for the current subdomain
            window_function_current = window_function.window_function_sigmoid_pou(test_points[:,0:1], test_points[:,1:2], i)

            # change the dtype of window function to i_dtype
            window_function_current = tf.cast(window_function_current, dtype=i_dtype)
            

            y_pred_window  =  y_pred_subdomain * window_function_current

            # y_pred_window *= tf.tanh(2.0*np.pi*(test_points[:, 0:1]))*tf.tanh(2.0*np.pi*(test_points[:, 1:2]))*tf.tanh(2.0*np.pi*(1.0 - test_points[:, 0:1]))*tf.tanh(2.0*np.pi*(1.0 - test_points[:, 1:2]))
            # y_pred_window *= models_subdomain[i].ansatz_kernel(models_subdomain[i].ansatz_slope)(test_points[:, 0:1]) * models_subdomain[i].ansatz_kernel(models_subdomain[i].ansatz_slope)(test_points[:, 1:2]) 
            # y_pred_window *=  tf.sin(2.0*np.pi*test_points[:, 0:1])*tf.sin(2.0*np.pi*test_points[:, 1:2])*y_pred_window

            if plot_subomain_solution:
                plot_contour(x = test_points[:,0].numpy().reshape(i_n_test_points_x, i_n_test_points_y), \
                                y = test_points[:,1].numpy().reshape(i_n_test_points_x, i_n_test_points_y),\
                                z = y_pred_window.numpy().reshape(i_n_test_points_x, i_n_test_points_y),\
                                output_path = i_output_path, filename= f"predictions_window_{epoch}_{i}", title = "")
                
                # save the file
                np.savetxt(i_output_path + f"/predictions_window_{epoch}_{i}.txt", y_pred_window.numpy().reshape(i_n_test_points_x, i_n_test_points_y))

            y_pred += y_pred_window

        y_pred *=  tf.tanh(2.0*np.pi*test_points[:, 0:1])*tf.tanh(2.0*np.pi*test_points[:, 1:2])*tf.tanh(2.0*np.pi*(1.0 - test_points[:, 0:1]))*tf.tanh(2.0*np.pi*(1.0 - test_points[:, 1:2]))
        # Multiply boundary condition ansatz functions with the predictions
        # y_pred = tf.tanh((test_points[:, 0:1]))*tf.tanh((test_points[:, 1:2]))*tf.tanh((test_points[:, 0:1]-1.0))*tf.tanh((test_points[:, 1:2]-1.0)) * y_pred


        # reshape the predictions for plotting
        y_pred = tf.reshape(y_pred, [i_n_test_points_x, i_n_test_points_y])

        X = test_points[:,0].numpy().reshape(i_n_test_points_x, i_n_test_points_y)
        Y = test_points[:,1].numpy().reshape(i_n_test_points_x, i_n_test_points_y)
        Y_Pred_Matrix = y_pred.numpy().reshape(i_n_test_points_x, i_n_test_points_y)
        Y_Exact_Matrix = y_exact.reshape(i_n_test_points_x, i_n_test_points_y)

        # plot the predictions
        plot_contour(x = X, y = Y, z = Y_Pred_Matrix, output_path = i_output_path, filename= "predictions"+str(epoch), title = "Predictions")
        plot_contour(x = X, y = Y, z = Y_Exact_Matrix, output_path = i_output_path, filename= "exact_solution", title = "Exact Solution")
        plot_contour(x = X, y = Y, z = np.abs(Y_Pred_Matrix - Y_Exact_Matrix), output_path = i_output_path, filename= "error"+str(epoch), title = "Error")

        # flatten the Y_exact matrix and the Y_pred matrix
        y_exact_flatten = Y_Exact_Matrix.flatten()
        y_pred_flatten = Y_Pred_Matrix.flatten()

        # Calculate L1 and L2 error
        l1_error = np.mean(np.abs(y_exact_flatten - y_pred_flatten))
        l2_error = np.sqrt(np.mean((y_exact_flatten - y_pred_flatten)**2))
        linf_error = np.max(np.abs(y_exact_flatten - y_pred_flatten))

        console.print(f"[bold]L1 Error = [/bold] {l1_error:.3e}", end=" ")
        console.print(f"[bold]L2 Error = [/bold] {l2_error:.3e}", end=" ")
        console.print(f"[bold]Linf Error = [/bold] {linf_error:.3e}")

        
        return l1_error, l2_error, linf_error
    
    # ---------------------------------------------------------------#
    # ------------- PRE TRAINING INITIALISATIONS ------------------  #
    # ---------------------------------------------------------------#
    num_epochs = i_epochs  # num_epochs
    progress_bar = tqdm(total=num_epochs, desc='Training', unit='epoch', bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}", colour="green", ncols=100)
    loss_array = {}   # total loss
    for i in range(len(blocks_in_domain)):
        loss_array[i] = []

    test_loss_array = [] # test loss
    time_array_train = []   # time per epoc
    time_array_pretrain = [] # time per pretrain step
    time_array_update = [] # time per update step
    # beta - boundary loss parameters
    beta = tf.constant(i_beta, dtype=i_dtype)

    x_min_global = tf.constant(i_x_min, dtype=i_dtype)
    x_max_global = tf.constant(i_x_max, dtype=i_dtype)
    y_min_global = tf.constant(i_y_min, dtype=i_dtype)
    y_max_global = tf.constant(i_y_max, dtype=i_dtype)

    # ---------------------------------------------------------------#
    # ------------- TRAINING LOOP ---------------------------------- #
    # ---------------------------------------------------------------#
    # generate a dict of loss array
    loss_array = {}   # total loss
    for i in range(len(blocks_in_domain)):
        loss_array[i] = []
    
    combined_loss_array = []

    # Arrays to save solution
    l1_error_array = []
    l2_error_array = []
    linf_error_array = []

    
    scheduler_num = 1
    trained_domains = []
    total_epochs = i_scheduling_freq*len(scheduler_list[scheduler_num])
    for epoch in range(total_epochs):   

        if epoch==0:
            active_domains = scheduler_list[scheduler_num]
            trained_domains = active_domains
            fixed_domains = []
        
        # pretrain step
        # for i in range(len(blocks_in_domain)):
        time_val = time.time()
        # for i in range(len(blocks_in_domain)):

        for i in active_domains:
            val, grad_x, grad_y, grad_xx, grad_yy  =  models_subdomain[i].pre_train_step(domain_decomposition = decomposed_domain, datahandler = datahandler[i])
            datahandler[i].reset_overlap_values()
            datahandler[i].update_current_domain_values(values = val, grad_x = grad_x, grad_y = grad_y, grad_xx = grad_xx, grad_yy = grad_yy)
        time_array_pretrain.append(time.time() - time_val)


        # gradient update step
        time_val = time.time()
        update_overlap_values_scheduled(domain_decomposition=decomposed_domain, datahandler=datahandler, fespace=fespace, active_domains=active_domains, fixed_domains=fixed_domains, trained_domains=trained_domains)
        time_array_update.append(time.time() - time_val)


        # train step
        # for i in range(len(blocks_in_domain)):
        time_val = time.time()
        for i in active_domains:
        # for i in range(len(blocks_in_domain)):
            loss = models_subdomain[i].train_step(beta=beta, bilinear_params_dict=bilinear_params_dict[i], 
                                                    overlap_val = datahandler[i].current_domain_overlap_values,
                                                    overlap_grad_x = datahandler[i].current_domain_overlap_grad_x,
                                                    overlap_grad_y = datahandler[i].current_domain_overlap_grad_y,
                                                    overlap_grad_grad_x = datahandler[i].current_domain_overlap_grad_xx,
                                                    overlap_grad_grad_y = datahandler[i].current_domain_overlap_grad_yy)
            # loss_array[i].append(loss["loss"])
        time_array_train.append(time.time() - time_val)
        progress_bar.update(1)

        if epoch % 1000 == 0 or epoch == num_epochs -1:
            if epoch == num_epochs -1:
                plot_subomain_solution_val=True
            else:
                plot_subomain_solution_val=False

            l1_error, l2_error, linf_error = predict_and_compute_loss_scheduled(domain = domain, blocks_in_domain = blocks_in_domain, models_subdomain = models_subdomain, \
                                        test_points = test_points, y_exact = y_exact, window_function = window_function, \
                                        i_dtype = i_dtype, i_n_test_points_x = i_n_test_points_x, i_n_test_points_y = i_n_test_points_y, \
                                        i_output_path = i_output_path, trained_domains=trained_domains, epoch=epoch, plot_subomain_solution=plot_subomain_solution_val)
            
            # Append the errors to the array
            l1_error_array.append(l1_error)
            l2_error_array.append(l2_error)
            linf_error_array.append(linf_error)
            # plot loss 
            for i in range(len(blocks_in_domain)):
                # plot_loss_function(loss_array[i], output_path = i_output_path, filename = f"loss_function_{i}")
                plot_loss_function(loss_array[i], output_path = i_output_path)


            # Sum all the losses into combined loss array
            # create combined loss array with zeros of length of loss array
            combined_loss_array = np.zeros(len(loss_array[0]))
            # for i in range(len(blocks_in_domain)):
            #     combined_loss_array += np.array(loss_array[i])
            
            # # plot the combined loss
            # plot_loss_function(combined_loss_array, output_path = i_output_path, filename = "combined_loss_function")
        
        if epoch % i_scheduling_freq == 0 and epoch != 0:
            scheduler_num, active_domains, trained_domains, fixed_domains = scheduling.obtain_next(scheduler_num, scheduler_list, active_domains, trained_domains, fixed_domains)


            
    progress_bar.close()

    exit()

    params_dict = {}
    params_dict['n_cells'] = fespace.n_cells

    # get the input data for the PDE
    train_dirichlet_input, train_dirichlet_output = datahandler.get_dirichlet_input()

    # get bilinear parameters
    # this function will obtain the values of the bilinear parameters from the model
    # and convert them into tensors of desired dtype
    bilinear_params_dict = datahandler.get_bilinear_params_dict_as_tensors(get_bilinear_params_dict)

    model = DenseModel(
        layer_dims=[2, 30, 30, 30, 1],
        learning_rate_dict=i_learning_rate_dict,
        params_dict=params_dict,
        loss_function=pde_loss_poisson,
        input_tensors_list=[datahandler.x_pde_list, train_dirichlet_input, train_dirichlet_output],
        orig_factor_matrices=[
            datahandler.shape_val_mat_list,
            datahandler.grad_x_mat_list,
            datahandler.grad_y_mat_list,
        ],
        force_function_list=datahandler.forcing_function_list,
        tensor_dtype=i_dtype,
        use_attention=i_use_attention,
        activation=i_activation,
        hessian=False,
    )

    test_points = domain.get_test_points()
    print(f"[bold]Number of Test Points = [/bold] {test_points.shape[0]}")
    y_exact = exact_solution(test_points[:, 0], test_points[:, 1])

    # save points for plotting
    X = test_points[:, 0].reshape(i_n_test_points_x, i_n_test_points_y)
    Y = test_points[:, 1].reshape(i_n_test_points_x, i_n_test_points_y)
    Y_Exact_Matrix = y_exact.reshape(i_n_test_points_x, i_n_test_points_y)

    # plot the exact solution
    plot_contour(
        x=X,
        y=Y,
        z=Y_Exact_Matrix,
        output_path=i_output_path,
        filename="exact_solution",
        title="Exact Solution",
    )

    num_epochs = i_epochs  # num_epochs
    progress_bar = tqdm(
        total=num_epochs,
        desc='Training',
        unit='epoch',
        bar_format="{l_bar}{bar:40}{r_bar}{bar:-10b}",
        colour="green",
        ncols=100,
    )
    loss_array = []  # total loss
    test_loss_array = []  # test loss
    time_array = []  # time per epoc
    # beta - boundary loss parameters
    beta = tf.constant(i_beta, dtype=i_dtype)

    # ---------------------------------------------------------------#
    # ------------- TRAINING LOOP ---------------------------------- #
    # ---------------------------------------------------------------#
    for epoch in range(num_epochs):

        # Train the model
        batch_start_time = time.time()
        loss = model.train_step(beta=beta, bilinear_params_dict=bilinear_params_dict)
        elapsed = time.time() - batch_start_time

        # print(elapsed)
        time_array.append(elapsed)

        loss_array.append(loss['loss'])

        # ------ Intermediate results update ------ #
        if (epoch + 1) % i_update_console_output == 0 or epoch == num_epochs - 1:
            y_pred = model(test_points).numpy()
            y_pred = y_pred.reshape(-1)

            error = np.abs(y_exact - y_pred)

            # get errors
            (
                l2_error,
                linf_error,
                l2_error_relative,
                linf_error_relative,
                l1_error,
                l1_error_relative,
            ) = compute_errors_combined(y_exact, y_pred)

            loss_pde = float(loss['loss_pde'].numpy())
            loss_dirichlet = float(loss['loss_dirichlet'].numpy())
            total_loss = float(loss['loss'].numpy())

            # Append test loss
            test_loss_array.append(l1_error)

            console.print(f"\nEpoch [bold]{epoch+1}/{num_epochs}[/bold]")
            console.print("[bold]--------------------[/bold]")
            console.print("[bold]Beta : [/bold]", beta.numpy(), end=" ")
            console.print(
                f"Variational Losses || Pde Loss : [red]{loss_pde:.3e}[/red] Dirichlet Loss : [red]{loss_dirichlet:.3e}[/red] Total Loss : [red]{total_loss:.3e}[/red]"
            )
            console.print(
                f"Test Losses        || L1 Error : {l1_error:.3e} L2 Error : {l2_error:.3e} Linf Error : {linf_error:.3e}"
            )

            plot_results(
                loss_array,
                test_loss_array,
                y_pred,
                X,
                Y,
                Y_Exact_Matrix,
                i_output_path,
                epoch,
                i_n_test_points_x,
                i_n_test_points_y,
            )

        progress_bar.update(1)

    # Save the model
    model.save_weights(str(Path(i_output_path) / "model_weights"))

    # print the Error values in table
    print_table(
        "Error Values",
        ["Error Type", "Value"],
        [
            "L2 Error",
            "Linf Error",
            "Relative L2 Error",
            "Relative Linf Error",
            "L1 Error",
            "Relative L1 Error",
        ],
        [l2_error, linf_error, l2_error_relative, linf_error_relative, l1_error, l1_error_relative],
    )

    # print the time values in table
    print_table(
        "Time Values",
        ["Time Type", "Value"],
        [
            "Time per Epoch(s) - Median",
            "Time per Epoch(s) IQR-25% ",
            "Time per Epoch(s) IQR-75% ",
            "Mean (s)",
            "Epochs per second",
            "Total Train Time",
        ],
        [
            np.median(time_array),
            np.percentile(time_array, 25),
            np.percentile(time_array, 75),
            np.mean(time_array),
            int(i_epochs / np.sum(time_array)),
            np.sum(time_array),
        ],
    )

    # save all the arrays as numpy arrays
    np.savetxt(str(Path(i_output_path) / "loss_function.txt"), np.array(loss_array))
    np.savetxt(str(Path(i_output_path) / "prediction.txt"), y_pred)
    np.savetxt(str(Path(i_output_path) / "exact.txt"), y_exact)
    np.savetxt(str(Path(i_output_path) / "error.txt"), error)
    np.savetxt(str(Path(i_output_path) / "time_per_epoch.txt"), np.array(time_array))
