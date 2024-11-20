"""

"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

class WindowFunction():
    """
    Class representing a window function used for partition of unity.

    Attributes:
        None

    Methods:
        kernel: Returns the kernel of the window function.
        window_function_sum: Returns the sum of the window functions.
        window_function_pou: Returns the partition of unity window function.
        plot_window_function: Plots the window function.
    """
    def __init__(self, x_mean_list, x_span_list, y_mean_list, y_span_list, kernel_name="rbf", scaling_factor=75, overlap_dim=0.5):
        """
        Constructor for the WindowFunction class.

        Args:
            x_mean_list: List of x_mean values of the blocks.
            x_span_list: List of x_span values of the blocks.
            y_mean_list: List of y_mean values of the blocks.
            y_span_list: List of y_span values of the blocks.
        """
        self.x_mean = x_mean_list
        self.x_span = x_span_list
        self.y_mean = y_mean_list
        self.y_span = y_span_list
        self.kernel_name = kernel_name
        self.scaling_factor = scaling_factor
        self.overlap_dim = overlap_dim

        if self.kernel_name not in ['cosine', 'sigmoid', 'rbf']:
            raise ValueError("Invalid kernel name. The kernel name should be one of the following: 'cosine', 'sigmoid', 'rbf'")


        self.x_min = [mean - span*0.5 for mean, span in zip(self.x_mean, self.x_span)]
        self.x_max = [mean + span*0.5 for mean, span in zip(self.x_mean, self.x_span)]
        self.y_min = [mean - span*0.5 for mean, span in zip(self.y_mean, self.y_span)]
        self.y_max = [mean + span*0.5 for mean, span in zip(self.y_mean, self.y_span)]

    
    def sigmoid_kernel(self, a, b):
        tol = 1e-10
        clamp = lambda f: tf.clip_by_value(f, tol, np.inf)
        kernel = lambda x: clamp(clamp(tf.sigmoid((x-(a))/self.scaling_factor))*clamp(tf.sigmoid(((b)-x)/self.scaling_factor)))
        return kernel
    

    # def kernel(self, block_mean, block_span):
    #     """
    #     This function returns the cosine kernel.

    #     Args:
    #         block_mean: Mean of the kernel.
    #         block_span: Span of the kernel.
    #     """

    #     tol = 1e-10
    #     clamp = lambda f: tf.clip_by_value(f, tol, np.inf)

    #     if self.kernel_name == 'cosine':
    #         kernel = lambda x: clamp(clamp(tf.square(1 + tf.cos(np.pi * ((x - block_mean) / (self.scaling_factor*block_span))))/4.0))
    #     elif self.kernel_name == 'sigmoid':
    #         kernel = lambda x: clamp(clamp(tf.sigmoid((x-(block_mean-block_span))/self.scaling_factor))*clamp(tf.sigmoid(((block_mean+block_span)-x)/self.scaling_factor)))
    #     elif self.kernel_name == 'rbf':
    #         kernel = lambda x: clamp(tf.exp(-1.0*self.scaling_factor*tf.square((x - block_mean))))
            
    #     else:
    #         raise ValueError("Invalid kernel name. The kernel name should be one of the following: 'cosine', 'sigmoid', 'rbf'")
        
    #     return kernel
    
    # def kernel_sigmoid(self, a, b):
    #     tol = 1e-10
    #     clamp = lambda f: tf.clip_by_value(f, tol, np.inf)
    #     kernel = lambda x: clamp(clamp(tf.sigmoid((x-(a))/self.scaling_factor))*clamp(tf.sigmoid(((b)-x)/self.scaling_factor)))
    #     return kernel

    def calc_a_b_x(self, block_id):
        a = (self.x_min[block_id] + (self.x_min[block_id] + self.overlap_dim))/2
        b = (self.x_max[block_id] + (self.x_max[block_id] - self.overlap_dim))/2
        return a, b
    def calc_a_b_y(self, block_id):
        a = (self.y_min[block_id] + (self.y_min[block_id] + self.overlap_dim))/2
        b = (self.y_max[block_id] + (self.y_max[block_id] - self.overlap_dim))/2
        return a, b
    
    def window_function_sigmoid(self, x, y, block_id):
        a_x, b_x = self.calc_a_b_x(block_id)
        a_y, b_y = self.calc_a_b_y(block_id)
        window_func = np.where((x < self.x_min[block_id]) | (x > self.x_max[block_id]) | (y < self.y_min[block_id]) | (y > self.y_max[block_id]), 0.0, self.sigmoid_kernel(a_x, b_x)(x) * self.sigmoid_kernel(a_y, b_y)(y))
        return window_func
    
    def window_function_sigmoid_sum(self, x, y):
        window_sum = 0
        for i in range(len(self.x_mean)):
            a_x, b_x = self.calc_a_b_x(i)
            a_y, b_y = self.calc_a_b_y(i)
            window_sum += self.sigmoid_kernel(a_x, b_x)(x) * self.sigmoid_kernel(a_y, b_y)(y)
        return window_sum
    
    def window_function_sigmoid_pou(self, x, y, block_id):
        a_x, b_x = self.calc_a_b_x(block_id)
        a_y, b_y = self.calc_a_b_y(block_id)
        window_sum = self.window_function_sigmoid_sum(x, y)
        window_pou = np.where((x < self.x_min[block_id]) | (x > self.x_max[block_id] ) | (y < self.y_min[block_id]) | (y > self.y_max[block_id]), 0.0, ((self.sigmoid_kernel(a_x, b_x)(x) * self.sigmoid_kernel(a_y, b_y)(y)) ))
        return window_pou/ window_sum

    
    def kernel_sum(self, x, y):
        """
        This function returns the sum of the window functions.

        Args:
            x: x-coordinate of the point.
            y: y-coordinate of the point.
        """
        kernel_sum = 0
        for i in range(len(self.x_mean)):
            kernel_sum += self.kernel(self.x_mean[i], self.x_span[i])(x) * self.kernel(self.y_mean[i], self.y_span[i])(y)
        # kernel_sum = 1.0
        return kernel_sum
    
    def window_function_pou(self, x, y, block_id):
        """
        This function returns the partition of unity window function.

        Args:
            x: x-coordinate of the point.
            y: y-coordinate of the point.
        """
        # if x is less than x_min or greater than x_max, return 0, else return the kernel value
        window_sum = self.kernel_sum(x, y)
        window_pou = np.where((x < self.x_min[block_id]) | (x > self.x_max[block_id] ) | (y < self.y_min[block_id]) | (y > self.y_max[block_id]), 0.0, ((self.kernel(self.x_mean[block_id], self.x_span[block_id])(x) * self.kernel(self.y_mean[block_id], self.y_span[block_id])(y)) ))
        
        # window_pou = self.kernel(self.x_mean[block_id], self.x_span[block_id])(x) * self.kernel(self.y_mean[block_id], self.y_span[block_id])(y) / window_sum
        return window_pou
    
    def plot_window_function(self, block_id, output_path="./"):
        """
        This function plots the window function.

        Args:
            block_id: Block id of the window function.
        """
        #create new directory in output_path as window_functions
        output_path = Path(output_path) / "window_functions"

        output_path.mkdir(parents=True, exist_ok=True)

        x = np.linspace(0, 1, 1000)
        y = np.linspace(0, 1, 1000)
        X, Y = np.meshgrid(x, y)
        Z = self.window_function_sigmoid_pou(X, Y, block_id)
        plt.axhline(y=self.y_min[block_id], color='k', linestyle='--', linewidth=0.5)
        plt.axhline(y=self.y_max[block_id], color='k', linestyle='--', linewidth=0.5)
        plt.axvline(x=self.x_min[block_id], color='k', linestyle='--', linewidth=0.5)
        plt.axvline(x=self.x_max[block_id], color='k', linestyle='--', linewidth=0.5)
        plt.contourf(X, Y, Z, levels=100, cmap='jet')
        plt.colorbar()
        plt.savefig(str(Path(output_path) / f"window_function_{block_id}.png"))
        plt.close()

    
        
    




        
        