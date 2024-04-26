# Using FastVPINNs for solving forward and inverse problems.

This folder contains annotated examples on solving both forward and inverse problems. The folder is divided into two sub-folders:
1. [`forward_problems_2d`](#solving-forward-problems)
2. [`inverse_problems_2d`](#solving_inverse_problems)

______________________

## Solving forward problems
FastVPINNs can solve several PDEs on different types of meshes using either soft or hard boundary constraints. In the sub-folder `forward_problems_2d`, you will find the files and markdown documentation to solve the following problems: 
1. [Convection-Diffusion equation on a circular mesh](./forward_problems_2d/complex_mesh/cd2d/README.md)
2. [Convection-Diffusion equation on gear mesh](./forward_problems_2d/complex_mesh/cd2d_gear/README.md)
3. [Helmholtz equation on a circular mesh](./forward_problems_2d/complex_mesh/helmholtz2d/README.md)
4. [Poisson equation on a circular mesh](./forward_problems_2d/complex_mesh/poisson2d/README.md)
5. [Poisson equation with hard boundary constraints](./forward_problems_2d/hard_boundary_constraints/poisson_2d/README.md)
6. [Poisson equation on a uniform mesh](./forward_problems_2d/uniform_mesh/poisson_2d/README.md)
7. [Helmholtz equation on a uniform mesh](./forward_problems_2d/uniform_mesh/helmholtz_2d/README.md)

## Solving inverse problems
FastVPINNs can solve parameter estimation problems for different PDEs. These parameters could be uniform on a domain or spatially dependent. In the `inverse_problems_2d` sub-folder, you will find the following examples:
1. [Estimating a uniform diffusion parameter for the Poisson problem](./inverse_problems_2d/const_inverse_poisson2d/README.md)
2. [Estimating a spatially dependent diffusion parameter for the Convection-Diffusion problem](./inverse_problems_2d/domain_inverse_cd2d/README.md)

____________________

This page is maintained by [Thivin](https://github.com/thivinanandh) and [Divij](https://divijghose.github.io). To add more examples to this list, please create a pull request.