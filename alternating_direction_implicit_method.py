# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Alternating Direction Implicit Method Applied to the 3D Wave Equation
# Both the simple explicit and simple implicit methods used in `simple_explicit_implicit_methods.ipynb` can be applied to higher-dimensional problems. However, they become very computationally expensive<sub>[1]</sub>. This motivates the alternating direction implicit method, which combines the simple explicit and simple implicit methods to produce finite difference discretizations corresponding to efficiently-solvable tridiagonal matrix equations.
#
# Consider the wave equation:
#
# $$ \frac{\partial^{2} u}{\partial t^{2}} = c^{2} \nabla^{2} u $$
#
# Where $\nabla^{2}$ is the spatial Laplacian operator. In three dimensions, and letting $c^{2} = 1$, the PDE becomes the following:
#
# $$ \frac{\partial^{2} u}{\partial t^{2}} = \frac{\partial^{2} u}{\partial x^{2}} + \frac{\partial^{2} u}{\partial y^{2}} + \frac{\partial^{2} u}{\partial z^{2}} $$
#
# There is now a choice to be made regarding finite difference discretizations of the spatial partial derivatives. Either the implicit or explicit methods could be employed to approximate a second derivative in an arbitrary spatial dimension. Superscripts denote a time step and subscripts denote a spatial node for the remainder of this document. The explicit centered-difference discretization is taken at the current time step, $l$, so that the values of $u$ are known thanks to an initial condition:
#
# $$ \frac{\partial^{2} u}{\partial x^{2}} \approx \frac{u^{l}_{i - 1, \, j, \, k} - 2u^{l}_{i, \, j, \, k} + u^{l}_{i + 1, \, j, \, k}}{(\Delta x)^{2}} $$
#
# And the implicit centered-difference discretization is taken at the next time step, $l + 1$. Values of $u$ are <i>unknown</i> in this discretization:
#
# $$ \frac{\partial^{2} u}{\partial x^{2}} \approx \frac{u^{l + 1}_{i - 1, \, j, \, k} - 2u^{l + 1}_{i, \, j, \, k} + u^{l + 1}_{i + 1, \, j, \, k}}{(\Delta x)^{2}} $$
#
# Rather than solving for the future values of $u$ in all dimensions at once, the implicit method discretization can be applied to one dimension at a time and the resulting tridiagonal matrix equation solved for values at a partial time step in the future<sub>[1]</sub>. Since there are three dimensions that need solving, this partial time step is chosen to be $1/3$ so that a whole step has elapsed after the third dimension is solved.
#
# $$ \frac{u^{l - 1/3}_{i, \, j, \, k} - 2u^{l}_{i, \, j, \, k} + u^{l + 1/3}_{i, \, j, \, k}}{(\Delta t)^{2}} = \overbrace{\frac{u^{l + 1/3}_{i - 1, \, j, \, k} - 2u^{l + 1/3}_{i, \, j, \, k} + u^{l + 1/3}_{i + 1, \, j, \, k}}{(\Delta x)^{2}}}^{\text{Implicit in the x dimension}} + \underbrace{\frac{u^{l}_{i, \, j - 1, \, k} - 2u^{l}_{i, \, j, \, k} + u^{l}_{i, \, j + 1, \, k}}{(\Delta y)^{2}} + \frac{u^{l}_{i, \, j, \, k - 1} - 2u^{l}_{i, \, j, \, k} + u^{l}_{i, \, j, \, k + 1}}{(\Delta z)^{2}}}_{\text{Explicit in other dimensions}} $$
#
# The expression can be simplified with the establishment of a uniform grid where $\Delta d = \Delta x = \Delta y = \Delta z$.
#
# $$ \frac{u^{l - 1/3}_{i, \, j, \, k} - 2u^{l}_{i, \, j, \, k} + u^{l + 1/3}_{i, \, j, \, k}}{(\Delta t)^{2}} = \frac{1}{( \Delta d )^{2}} \left ( u^{l + 1/3}_{i - 1, \, j, \, k} - 2u^{l + 1/3}_{i, \, j, \, k} + u^{l + 1/3}_{i + 1, \, j, \, k} + u^{l}_{i, \, j - 1, \, k} - 2u^{l}_{i, \, j, \, k} + u^{l}_{i, \, j + 1, \, k} + u^{l}_{i, \, j, \, k - 1} - 2u^{l}_{i, \, j, \, k} + u^{l}_{i, \, j, \, k + 1} \right ) $$
#
# Defining $\lambda \equiv ( \Delta d / \Delta t )^{2}$, combining terms, and isolating the unknown future values on the left side yields the following:
#
# $$ -u^{l + 1/3}_{i - 1, \, j, \, k} + ( \lambda + 2 ) u^{l + 1/3}_{i, \, j, \, k} - u^{l + 1/3}_{i + 1, \, j, \, k} = 2 ( \lambda - 2 ) u^{l}_{i, \, j, \, k} - \lambda u^{l - 1/3}_{i, \, j, \, k} + u^{l}_{i, \, j - 1, \, k} + u^{l}_{i, \, j + 1, \, k} + u^{l}_{i, \, j, \, k - 1} + u^{l}_{i, \, j, \, k + 1} $$
#
# Which, for a domain of $0$ to $n$ nodes in the $x$ dimension, corresponds to the tridiagonal matrix equation below. Notice that there are equations written only for nodes $1$ to $n - 1$ because the discretization can only be applied to interior nodes. $u^{l + 1/3}_{0, \, j, \, k}$ and $u^{l + 1/3}_{n, \, j, \, k}$ in the right hand side vector are the future values of the exterior nodes, and should be set according to the problem's boundary conditions.
#
# $$ 
# \begin{pmatrix} 
# ( \lambda + 2 ) & -1 & & & 0 \\
# -1 & ( \lambda + 2 ) & -1 & & \\
# & \ddots & \ddots & \ddots & \\
# & & -1 & ( \lambda + 2 ) & -1 \\
# 0 & & & -1 & ( \lambda + 2 )
# \end{pmatrix} 
# \begin{pmatrix} u^{l + 1/3}_{1, \, j, \, k} \\
# u^{l + 1/3}_{2, \, j, \, k} \\
# \vdots \\
# u^{l + 1/3}_{n - 2, \, j, \, k} \\
# u^{l + 1/3}_{n - 1, \, j, \, k}
# \end{pmatrix}
# =
# \begin{pmatrix}
# u^{l + 1/3}_{0, \, j, \, k} + 2 ( \lambda - 2 ) u^{l}_{i, \, j, \, k} - \lambda u^{l - 1/3}_{i, \, j, \, k} + u^{l}_{i, \, j - 1, \, k} + u^{l}_{i, \, j + 1, \, k} + u^{l}_{i, \, j, \, k - 1} + u^{l}_{i, \, j, \, k + 1} \\
# 2 ( \lambda - 2 ) u^{l}_{i, \, j, \, k} - \lambda u^{l - 1/3}_{i, \, j, \, k} + u^{l}_{i, \, j - 1, \, k} + u^{l}_{i, \, j + 1, \, k} + u^{l}_{i, \, j, \, k - 1} + u^{l}_{i, \, j, \, k + 1} \\
# \vdots \\
# 2 ( \lambda - 2 ) u^{l}_{i, \, j, \, k} - \lambda u^{l - 1/3}_{i, \, j, \, k} + u^{l}_{i, \, j - 1, \, k} + u^{l}_{i, \, j + 1, \, k} + u^{l}_{i, \, j, \, k - 1} + u^{l}_{i, \, j, \, k + 1} \\
# u^{l + 1/3}_{n, \, j, \, k} + 2 ( \lambda - 2 ) u^{l}_{i, \, j, \, k} - \lambda u^{l - 1/3}_{i, \, j, \, k} + u^{l}_{i, \, j - 1, \, k} + u^{l}_{i, \, j + 1, \, k} + u^{l}_{i, \, j, \, k - 1} + u^{l}_{i, \, j, \, k + 1}
# \end{pmatrix}
# $$
#
# Muliple matrix equations need to be solved for a single spatial dimension since location on the $y$ and $z$ axes is required to select specific values of $u$. If the domain is a cube, as it is in this problem, then matrix equations such as that above must be solved $( n - 1 )^{2}$ times for each dimension. When all $3 ( n - 1 )^{2}$ tridiagonal systems have been solved, the simulation is at time step $l + 1$, and the process can be repeated for the remaining times. This procedure is implemented in the following cells.

# %% [markdown]
# ## References
# [1] Chapra, S. C., &amp; Canale, R. P. (2015). Numerical Methods for Engineers (7th ed.). New York, NY: McGraw-Hill Education.
