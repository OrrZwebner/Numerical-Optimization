import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



class InteriorPointOptimizer:
    """
    class to perform interior point optimization.
    """

    def __init__(self,func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, max_outer_iter=1000 ,max_inner_iter=100):
        """
        Constructor for the InteriorPointOptimizer class.
        :param func: Objective function
        :param ineq_constraints: List of inequality constraints functions
        :param eq_constraints_mat (np.array): Matrix of equality constraints (Ax = b). If there are no equality constraints, set to None. 
        :param eq_constraints_rhs (np.array): Right hand side of the equality constraints (Ax = b). If there are no equality constraints, set to None.
        """
        self.f = func # Objective function
        self.ineq_constraints = ineq_constraints # List of inequality constraints
        self.eq_constraints_mat = eq_constraints_mat # Matrix of equality constraints
        self.eq_constraints_rhs = eq_constraints_rhs # Right hand side of the equality constraints
        self.x0 = x0 # Initial point
        self.max_outer_iter = max_outer_iter # Maximum number of outer iterations
        self.max_inner_iter = max_inner_iter # Maximum number of inner iterations



    def phi(self, x, hess_flag=True):
        """
        Compute the barrier function and its gradient and hessian in a given point x.
        :param constraints: List of constraint functions
        :param x: Current parameter values
        :param hess_flag: Flag to evaluate the Hessian matrix. default is True.
        :return: Barrier function value, gradient and hessian
        """
        phi, grad_phi, hessian_phi = 0, np.zeros_like(x), np.zeros((len(x), len(x))) # Initialize the barrier function, gradient and hessian to zero

        if hess_flag:
            for f in self.ineq_constraints: # Iterate over the the inequality constraint functions and compute the barrier function, gradient and hessian
                f_x, g_x, h_x = f(x, hess_flag=True) # Compute the function value, gradient and hessian at x for the current constraint
                if f_x >= 0:
                    return np.inf, np.inf, np.inf
                
                else:
                    phi -= np.log(-f_x) # Update the barrier function
                    grad_phi -= g_x / (-f_x)  # Compute the gradient of phi using the gradient of the inequality constraint divided by the value of the inequality constraint
                    hessian_phi += (np.outer(g_x, g_x) / (f_x ** 2)) + (h_x / f_x) # Update the hessian of phi using the hessian of the inequality constraint and the gradient of phi

            return phi, grad_phi, hessian_phi
        
        else: # If the flag to evaluate the Hessian matrix is False
            for f in self.ineq_constraints: # Iterate over the the inequality constraint functions and compute the barrier function, gradient and hessian
                f_x, g_x = f(x, hess_flag=False) # Compute the function value and gradient at x for the current constraint 
                if f_x >= 0: # If the value of the inequality constraint is greater than or equal to zero
                    return np.inf, np.inf # Return infinity for the barrier function and gradient
                
                else: # If the value of the inequality constraint is less than zero
                    phi -= np.log(-f_x) # Update the barrier function
                    grad_phi -= g_x / (-f_x) # Compute the gradient of phi using the gradient of the inequality constraint divided by the value of the inequality constraint
            # Return the barrier function and gradient of the barrier function (phi)
            
            return phi, grad_phi
    

    def modified_objective(self, x, t, hess_flag=True):
        """
        Computes the modified objective, gradient and hessian including the barrier term of the objective function.

        Parameters:
        x (np.ndarray): The point at which to evaluate the modified objective function.
        t (float): The barrier parameter.
        hess_flag (bool): Flag to evaluate the Hessian matrix. default is True.

        Returns:
        a tuple of the modified objective value, gradient and hessian at x.
        """
        if hess_flag:
            f_val, grad, hess = self.f(x, hess_flag=True) # Compute the value, gradient and hessian of the objective function at x
            phi_val, grad_phi, hess_phi = self.phi(x, hess_flag=True) # Compute the value, gradient and hessian of the barrier function at x
            return t * f_val + phi_val, t * grad + grad_phi, t * hess + hess_phi
        
        else:
            f_val, grad = self.f(x, hess_flag=False)
            phi_val, grad_phi = self.phi(x, hess_flag=False)
            return t * f_val + phi_val, t * grad + grad_phi

    


    def interior_pt(self, t=1, mu=10, epsilon=1e-6):
        """
        Minimize func subject to  ineq_constraints, and to the affine innequality defined by matrix eq_constraints_mat and the right hand side vector eq_constraints_rhs.
        Parameters:
        :param t: barrier parameter (float)
        :param mu: barrier parameter update factor (float)
        :param epsilon: tolerance (float)

        Returns:
        :return: path: list of tuples (iteration, objective value, point)
        :return: x: optimal point (np.array)
        :return: f: optimal value (float)
        """


        m = len(self.ineq_constraints) # number of inequality constraints

  
        outer_iter = 0 # number of iterations
        cur_x = self.x0.copy() # initial point
        prev_f = np.inf # Initialize the previous objective value as infinity
        cur_f , _ = self.f(cur_x, hess_flag=False) # Initialize the current objective value  

        x_path = self.x0.copy() # Initialize a copy of the initial point to store the path

        path = [(0, self.f(x_path)[0], x_path)] # list to store the path of the algorithm (iterations, objective values, x values)
        alpha = 1.0 # Initialize the step length

        # outer loop - iterate until the number of inequality constraints divided by t is less than the tolerance epsilon
        while m / t > epsilon:

            newton_decrement = 1.0 # Initialize the step length

            if outer_iter >  0: # If the number of inner iterations is greater than 0 - update the previous objective value
                prev_f = cur_f # Update the previous objective value
                cur_f , _ = self.f(cur_x, hess_flag=False) # Initialize the current objective value  
                
            inner_iter = 0 # number of iterations
            # inner loop - iterate until the change in the objective value and parameters is less than the tolerance, or the number of inner iterations is less than the maximum number of inner iterations

            while 0.5 * (newton_decrement ** 2) > epsilon and np.abs(prev_f - cur_f) > epsilon and inner_iter < self.max_inner_iter:  # stop criterion

                prev_f = cur_f # Update the previous objective value, gradient and hessian
                cur_f, cur_gradient, cur_hessian = self.modified_objective(cur_x, t) # Compute the value, gradient and hessian of the modified objective function at the current point
                newton_dir, newton_decrement = self.newton_step(cur_f, cur_gradient, cur_hessian, cur_x) # Compute the Newton step and decrement

                alpha = self.backtracking_line_search(cur_x, newton_dir, t) # Perform backtracking line search to find the step length 
                cur_x += alpha * newton_dir # Update the current point using the step length and the Newton direction
                inner_iter += 1 # Increment the number of iterations

                
            if inner_iter > 0 : # If the number of inner iterations is greater than 0  - avoid duplications in the path
                outer_iter += 1 # Increment the number of iterations
                x_path = cur_x.copy()
                path.append((outer_iter, self.f(x_path)[0], x_path)) # Append the current iteration, objective value and point to the path

            if m / t > epsilon: # If the number of inequality constraints divided by t is greater than the tolerance epsilon
                t *= mu

        # return the path, the optimal point x, the optimal value f, the number of  outer iterations
        return path, cur_x, self.f(cur_x)[0]


    def backtracking_line_search(self, x, direction, t, c1=0.01, beta=0.5):
        """
        Perform backtracking line search with the first Wolfe condition to find step length.
        :param x: Current parameter values
        :param direction: Search direction
        :param t: Penalty parameter
        :param c1: Constant for the first Wolfe condition
        :param beta: Constant for backtracking
        """
        # Initialize step length
        step_length = 1.0

        f_val, grad = self.modified_objective(x, t, hess_flag=False) # Compute function value and gradient
        f_step, __= self.modified_objective(x + step_length * direction, t, hess_flag=False) # Compute function value at new point

        # Perform backtracking line search with Wolfe condition (first condition)   
        while f_step > f_val + c1 * step_length * np.dot(grad, direction) and step_length > 1e-6:
            step_length *= beta # Reduce step length
            f_step, __= self.modified_objective(x + step_length * direction, t, hess_flag=False) # Compute function value at new point


        return step_length
    

    def newton_step(self, f, grad, hess, x_k, tol=1e-6):
        """
        Compute the Newton step for the function f according to the equality constraints matrix at the point x_k.
        :param f: Objective value at x_k
        :param grad: Gradient value of the objective function at x_k
        :param hess: Hessian value of the objective function at x_k
        :param x_k: Current parameter values
        :param tol: Tolerance for stopping criterion
        :return: Newton direction (np.array) and Newton decrement (float)
        """

        # Compute the Newton direction using the equality constraints matrix
        A = self.eq_constraints_mat # Matrix of equality constraints
        b = self.eq_constraints_rhs # Right hand side of the equality constraints


        if A is None: # If there are no equality constraints
            try:
                search_dir = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                print(f'Non singular Hessian matrix:\n{hess}')
                # assign 0 
                search_dir = np.zeros_like(x_k)

        
        else: # If there are equality constraints
            # Construct the block matrix for the interior point optimization
            matrix_block = np.block([[hess, A.T], [A, np.zeros((A.shape[0], A.shape[0]))]]) 
            vector_block = np.concatenate([-grad, np.zeros(A.shape[0])])

            # Solve the Newton system
            try:
                solution = np.linalg.solve(matrix_block, vector_block)
            except np.linalg.LinAlgError:
                print(f'Non singular Hessian matrix:\n{hess}')
                # assign 0 
                search_dir = np.zeros_like(x_k)
            search_dir = solution[:x_k.shape[0]] # Compute the search direction 

        # Compute the Newton decrement
        nt_decrement = np.sqrt(search_dir.T @ hess @ search_dir)   

        return search_dir, nt_decrement




