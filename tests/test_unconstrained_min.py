import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from examples import *
from src.unconstrained_min import unconstrained_min, backtracking_line_search
from src.utils import *

class TestUnconstrainedMin(unittest.TestCase):
    def test_minimization(self):
        x0 = np.array([1, 1]) # Initial point
        x0_rosnbrock = np.array([-1, 2]) # Initial point for Rosenbrock function
        obj_tol = 1e-12 # Objective tolerance
        param_tol = 1e-8 # Parameter tolerance
        max_iter = 100 # Maximum number of iterations
        max_iter_rosenbrock = 10000 # Maximum number of iterations for Rosenbrock function (higher due to slow convergence)
        x1_min=-2
        x1_max=2
        x2_min=-2
        x2_max=2
        levels=25

        # iterate over the functions in examples file and test the unconstrained_min function with both gradient_descent and newton methods
        for f in [quadratic_example_1, quadratic_example_2, quadratic_example_3, rosenbrock_function, linear_function, corner_triangles_function]:
            # Set the initial point based on the function
            if f == rosenbrock_function:
                result_gd = unconstrained_min(f, x0_rosnbrock, obj_tol, param_tol, max_iter_rosenbrock, 'gradient_descent') # Gradient Descent
                result_newton = unconstrained_min(f, x0_rosnbrock, obj_tol, param_tol, max_iter, 'newton') # Newton's method
                x2_max = 3 # Set the maximum value of x2 for Rosenbrock function for the plot

            else:
                result_gd = unconstrained_min(f, x0, obj_tol, param_tol, max_iter, 'gradient_descent') # Gradient Descent
                result_newton = unconstrained_min(f, x0, obj_tol, param_tol, max_iter, 'newton') # Newton's method

            if f == linear_function:
                levels = 25
                x1_min = -200
                x2_min = -300


            plot_contour(f, result_gd[3], result_newton[3], function_name=f.__name__, x1_min=x1_min, x1_max=x1_max, x2_min=x2_min, x2_max=x2_max, levels=levels)

            plot_function_values(result_gd[3], result_newton[3], function_name=f.__name__)




if __name__ == '__main__':
    unittest.main()
