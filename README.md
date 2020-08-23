# Finite Difference Methods for Partial Differential Equations

<img src="output/3d_wave.gif" width="384" height="384">

This repository contains Jupyter notebooks and Python scripts which I write as I learn about various finite difference methods for solving partial differential equations.

### View with Binder (Recommended)
The fastest way to view my work is with an interactive Binder environment using the links below. Please expect to wait around 30 seconds for each notebook to launch.

* [simple_explicit_implicit_methods.ipynb](https://mybinder.org/v2/gl/smgill%2Ffinite-difference-methods-for-pdes/master?filepath=simple_explicit_implicit_methods.ipynb): Simple explicit and simple implicit methods are used to solve the heat equation in one dimension modeling heat conduction in a long, thin, copper wire.
* [alternating_direction_implicit_method.ipynb](https://mybinder.org/v2/gl/smgill%2Ffinite-difference-methods-for-pdes/master?filepath=alternating_direction_implicit_method.ipynb): Some sources seem to suggest that the alternating direction implicit method (ADI) is only applicable to parabolic and elliptic PDEs. In this notebook, it is demonstrated that ADI can be applied to solving the wave equation--which is hyperbolic--in three dimensions.

### View Locally
You can clone this repository and run the Jupyter notebooks locally. You need to [install Jupyter](https://jupyter.org/install.html) or use an IDE which can work with IPython notebooks such as Visual Studio Code or PyCharm. If you are using pip to handle modules, you can install this project's dependencies with `pip install -r requirements.txt`.