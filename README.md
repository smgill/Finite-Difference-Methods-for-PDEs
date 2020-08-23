# Finite Difference Methods for Partial Differential Equations

![](output/3d_wave.gif "A numerical solution to the 3D wave equation found in alternating_direction_implicit_method.ipynb")

This repository contains [Jupyter notebooks](https://jupyter.org/) and python scripts which I write as I learn about various finite difference methods for solving partial differential equations. The fastest way to view my work is with [Binder](https://mybinder.org/) using the links below:

* ### [simple_explicit_implicit_methods.ipynb]()
Simple explicit and simple implicit methods are used to solve the heat equation in one dimension modeling heat conduction in a thin copper wire.
* ### [alternating_direction_implicit_method.ipynb]()
Some sources seem to suggest that the alternating direction implicit method (ADI) is only applicable to parabolic and elliptic PDEs. Previously explored simple explicit and simple implicit methods are combined in this notebook to demonstrate that ADI can be applied to solving the 3D wave equation, which is hyperbolic.

Alternatively, you can clone this repository and run the Jupyter notebooks or Python scripts locally. If you are handling modules with pip, you can install dependencies with `pip install -r requirements.txt`.