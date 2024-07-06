import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the system path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to the system path
# print th ecurrent directory of the file
# print(f"current directory: {os.path.dirname(os.path.abspath(__file__))}")



from examples import *
from src.constrained_min import InteriorPointOptimizer
from src.utils import *

class TestconstrainedMin(unittest.TestCase):
        
    def test_constrained_min(self):
        """
        Define the functions, constraints and initial points for the optimization problem.
        """
        f_list = [qp, lp] # List of objective functions
        f_titles_list = ['qp_func', 'lp_func'] # List of titles for the objective functions
        x0_list = [np.array([0.1, 0.2, 0.7]), np.array([0.5, 0.75])]  # List of initial points
        ineq_qp = [qp_ineq_constraint_1, qp_ineq_constraint_2, qp_ineq_constraint_3] # List of inequality constraints for the quadratic programming example
        ineq_lp = [lp_ineq_constraint_1, lp_ineq_constraint_2, lp_ineq_constraint_3, lp_ineq_constraint_4] # List of inequality constraints for the linear programming example
        ineq = [ineq_qp, ineq_lp] # List of lists of inequality constraints
        eq_qp = [qp_eq_constraint] # List of equality constraints for the quadratic programming example
        eq_lp = None # List of equality constraints for the linear programming example
        eq = [eq_qp, eq_lp] # List of lists of equality constraints. Assigned such that the right side of the equality is 0.
        qp_bol_list = [True, False] # List of booleans, True if the function is a quadratic, False if linear



        for f, x0, ineq, eq_qp, title, qp_bol in zip(f_list, x0_list, ineq, eq, f_titles_list, qp_bol_list):
            if qp_bol:
                self.run_qp(f, x0, ineq, eq_qp, title)
            else:
                self.run_lp(f, x0, ineq, eq_qp, title)

        

    
    def run_qp(self, f, x0, ineq, eq_qp, title):
        """
        Run quadratic programming example.
        :param f: Objective function
        :param x0: Initial point
        :param ineq: List of inequality constraints
        :param eq_qp: List of equality constraints
        :param title: Title of the function
        """
        # convert the equality constraints to matrix form (if not None). do it by iterate over every equality and make sure to check the dimension of the equality constraints.
        if eq_qp is not None:
            A, b =  eq_qp[0]() # get the left and right side of the first equality constraint
            # iterate over the rest of the equality constraints and concatenate them to the matrix A and the vector b
            for eq in eq_qp[1:]:
                if eq is not None:
                    A = np.concatenate((A, eq()[0]), axis=0) # concatenate the left side of the equality constraint
                    b = np.concatenate((b, eq()[1]), axis=0) # concatenate the right side of the equality constraint
        # Call the interior point optimizer
        minimizer = InteriorPointOptimizer(func=f, ineq_constraints=ineq, eq_constraints_mat=A, eq_constraints_rhs=b, x0=x0)
    
        path, x_final, f_val = minimizer.interior_pt()


        # Report:
        print("-----")
        print("Final Results for the Quadratic Programming Example:")
        f_val, _ = f(x_final)
        print(f"The final candidtae is: {x_final} and the objective value is: {f_val}")

        print("in-equality constraints:")
        for i, ineq_const in enumerate(ineq):
            ineq_final = ineq_const(x_final)[0] * (-1)
            print(f"#{i + 1} contraint: {ineq_final}")
        print("equality constraints:")
        for i, eq_const in enumerate(eq_qp):
            eq_final = eq_const()[0] @ x_final - eq_const()[1]
            print(f"#{i + 1} constraint: {eq_final[0][0]}")

        # Plot the path
        qp_plot_3d(path, title)
        

        # plot function value vs iteration number:
        plot_function_values(result_newton_path=path, function_name="Quadratic Programming Example")

    def run_lp(self, f, x0, ineq, eq_lp, title):
        """
        Run linear programming example.
        :param f: Objective function
        :param x0: Initial point
        :param ineq: List of inequality constraints
        :param eq_lp: List of equality constraints
        """
        # Call the interior point optimizer
        if eq_lp is not None: # Check if there are equality constraints
            A, b = eq_lp[0]() # Get the left and right side of the first equality constraint
            for eq in eq_lp[1:]:  # Iterate over the rest of the equality constraints and concatenate them to the matrix A and the vector b
                if eq is not None: # Check if the equality constraint is not None
                    A = np.concatenate((A, eq()[0]), axis=0) # Concatenate the left side of the equality constraint
                    b = np.concatenate((b, eq()[1]), axis=0) # Concatenate the right side of the equality constraint
        else:
            A, b = None, None
        minimizer = InteriorPointOptimizer(func=f, ineq_constraints=ineq, eq_constraints_mat=A, eq_constraints_rhs=b, x0=x0)
        path, x_final, f_val = minimizer.interior_pt()

        # Report:
        print("-----")
        print("Final Results for the Linear Programming Example:")
        f_val, _= f(x_final)
        print(f"The final candidate is: {x_final} and the objective value is: {f_val}")

        print("in-equality constraints:")
        for i, ineq_const in enumerate(ineq):
            ineq_final = ineq_const(x_final)[0]
            print(f"#{i + 1} constraint: {ineq_final}")

        # Plot the path
        lp_plot_2d(path, title)

        # plot function value vs iteration number:
        plot_function_values(result_newton_path=path, function_name="Linear Programming Example")




if __name__ == '__main__':
    unittest.main()

