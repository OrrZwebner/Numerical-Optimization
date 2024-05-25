
import numpy as np

def unconstrained_min(f, x0, obj_tol, param_tol, max_iter, method='gradient_descent'):
    """
    Perform unconstrained optimization using gradient descent or Newton's method.
    :param f: Objective function
    :param x0: Initial parameter values
    :param obj_tol: Tolerance for change in objective value
    :param param_tol: Tolerance for change in parameters
    :param max_iter: Maximum number of iterations
    :param method: Optimization method ('gradient_descent' or 'newton')
    """
    
    # Initialize variables
    x = x0
    iter_count = 0
    obj_change = float('inf')
    param_change = float('inf')
    path = [] # list to store tuples of iterations, objective values and x values

    # Perform optimization until convergence or maximum iterations 
    while iter_count < max_iter and obj_change > obj_tol and param_change > param_tol:
        
        # Compute search direction using gradient descent or Newton method
        if method == 'gradient_descent':
            f_val, grad = f(x, hess_flag=False) # Compute function value and gradient
            direction = -grad # Compute search direction using gradient descent

        elif method == 'newton':
            f_val, grad, hessian = f(x, hess_flag=True)

            if np.linalg.det(hessian) == 0:
                success = False
                print(f'Non singular Hessian matrix at iteration {iter_count}, Newton solver failed')
                f_val_new = f_val # Keep the same objective value
                break
            else:
                direction = -np.linalg.inv(hessian) @ grad # Compute search direction using Newton's method

        path.append((iter_count, f_val, x)) # Append current iteration and objective value to path

        # Perform line search with Wolfe condition and backtracking
        step_length = backtracking_line_search(f, x, direction)

        # Print current location and objective value
        print(f'Iteration {iter_count}: current location = {x}, current objective value = {f_val}')

        # Update parameters
        x_new = x + step_length * direction # Update parameters
        # f_val, _ = f(x, hess_flag=False) # Compute function value
        f_val_new, _ = f(x_new, hess_flag=False) # Compute function value at new point
        obj_change = abs(f_val_new - f_val) # Compute change in objective value
        param_change = np.linalg.norm(x_new - x) # Compute change in parameters
        x = x_new # Update parameters
        
        # Increment iteration count
        iter_count += 1


    path.append((iter_count, f_val_new, x)) # append the last iteration and objective value to the path
    
    if obj_change <= obj_tol or param_change <= param_tol:
        success = True  # Optimization converged
    else:
        success = False # Optimization did not converge (reached maximum iterations)   

    # if f.__name__ == 'rosenbrock_functi dam_change <= param_tol: {param_change <= param_tol}')

    print(f'Last iteration, {f.__name__}, {method}, iteration {iter_count}: current location = {x}, current objective value = {f_val}, success flag = {success}')

    # The algorithm returns the final location, final objective value and a success/failure Boolean flag. Your algorithm should enable access to the entire path of iterations and objective values when done (either return them or store them in your class) for later usage in visualization.
    return x, f_val_new, success, path



def backtracking_line_search(f, x, direction, c1=0.01, beta=0.5):
    """
    Perform backtracking line search with the first Wolfe condition to find step length.
    :param f: Objective function
    :param x: Current parameter values
    :param direction: Search direction
    :param alpha: Constant for the first Wolfe condition
    :param beta: Constant for backtracking
    """
    # Initialize step length
    step_length = 1.0

    f_val, grad = f(x, hess_flag=False) # Compute function value and gradient
    f_step, _ = f(x + step_length * direction) # Compute function value at new point

    # Perform backtracking line search with Wolfe condition (first condition)   
    while f_step > f_val + c1 * step_length * np.dot(grad, direction):
        step_length *= beta # Reduce step length
        f_step, _ = f(x + step_length * direction) # Compute function value at new point

    return step_length




