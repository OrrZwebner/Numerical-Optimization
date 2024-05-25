import numpy as np

def quadratic_example_1(x, hess_flag=False):
    """
    Quadratic function example 1
    :param x: Input vector
    :param hessian_evaluation: Flag to evaluate the Hessian matrix
    """
    Q = np.array([[1, 0], [0, 1]]) # Quadratic matrix
    f = np.dot(x.T, np.dot(Q, x)) # Quadratic function
    g = 2 * np.dot(Q, x)  # Gradient

    if hess_flag:
        h = 2 * Q
        return f, g, h
    else:
        return f, g


def quadratic_example_2(x, hess_flag=False):
    """
    Quadratic function example 2
    :param x: Input vector
    :param hessian_evaluation: Flag to evaluate the Hessian matrix
    """
    Q = np.array([[1, 0], [0, 100]])
    f = np.dot(x.T, np.dot(Q, x))
    g = 2 * np.dot(Q, x)

    if hess_flag:
        h = 2 * Q 
        return f, g, h
    
    else:
        return f, g

def quadratic_example_3(x, hess_flag=False):
    """
    Quadratic function example 3
    :param x: Input vector
    :param hessian_evaluation: Flag to evaluate the Hessian matrix
    """
    Q1 = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Q2 = np.array([[100, 0], [0, 1]])
    # Q3 = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Q3 = Q1.T
    Q = np.dot(Q1, np.dot(Q2, Q3))
    f = np.dot(x.T, np.dot(Q, x))
    g = 2 * np.dot(Q, x)

    if hess_flag:
        h = 2 * Q  # Hessian
        return f, g, h
    else:
        return f, g

def rosenbrock_function(x, hess_flag=False):

    """
    Rosenbrock function
    :param x: Input vector
    :param hessian_evaluation: Flag to evaluate the Hessian matrix
    """
    # Check input dimensions
    if len(x) != 2:
        raise ValueError("Rosenbrock function is only calculated for 2D input")
        return None
    
    f = 100 * ((x[1] - x[0]**2)**2) + ((1 - x[0]))**2 # Objective function
    g = np.array([-400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]), 200 * (x[1] - x[0]**2)]) # Gradient

    if hess_flag:
        h = np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]]) # Hessian
        return f, g, h 
    else:
        return f, g
    

def linear_function(x, hess_flag=False):
    """
    Linear function example f(x) = 2x_1 + 3x_2
    :param x: Input vector
    :param hessian_evaluation: Flag to evaluate the Hessian matrix
    """
    a = np.array([2, 3]) # Coefficients
    f = np.dot(a, x) # Linear function
    g = a # Gradient
    if hess_flag:
        h = np.zeros((2, 2))
        return f, g, h
    else:
        return f, g


def corner_triangles_function(x, hess_flag=False):
    """
    Example from Boyd's book, p.470, example 9.2
    :param x: Input vector
    :param hessian_evaluation: Flag to evaluate the Hessian matrix
    """
    f = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1) # Objective function
    g = np.array([np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) - np.exp(-x[0] - 0.1),
                  3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)]) # Gradient
    
    if hess_flag:
        h = np.array([[np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + np.exp(-x[0] - 0.1),
                       3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1)],
                      [3 * np.exp(x[0] + 3 * x[1] - 0.1) - 3 * np.exp(x[0] - 3 * x[1] - 0.1),
                       9 * np.exp(x[0] + 3 * x[1] - 0.1) + 9 * np.exp(x[0] - 3 * x[1] - 0.1)]]) # Hessian
        
        return f, g, h
    
    else:
        return f, g
    

