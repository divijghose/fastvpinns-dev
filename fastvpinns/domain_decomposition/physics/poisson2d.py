"""
This file `poisson2d.py` is implementation of our efficient tensor-based loss calculation for poisson equation

Author: Thivin Anandh D

Date: 21/Sep/2023

History: Initial implementation

Refer: https://arxiv.org/abs/2404.12063
"""

import tensorflow as tf


# PDE loss function for the poisson problem
@tf.function
def pde_loss_poisson(
    forcing_function,
    pred_val,
    pred_grad_x_nn,
    pred_grad_y_nn,
    pred_grad_grad_x_nn,
    pred_grad_grad_y_nn,
):  # pragma: no cover
    """
    This method returns the loss for the Poisson Problem of the PDE
    """
    # ∫du/dx. dv/dx dΩ
    pde_diffusion_x = -1.0*pred_grad_grad_x_nn
    pde_diffusion_y = -1.0*pred_grad_grad_y_nn


    # eps * ∫ (du/dx. dv/dx + du/dy. dv/dy) dΩ
    pde_diffusion = (pde_diffusion_x + pde_diffusion_y)

    residual_matrix = pde_diffusion - forcing_function
    

    residual_cells = tf.reduce_mean(tf.square(residual_matrix), axis=0)

    return residual_cells