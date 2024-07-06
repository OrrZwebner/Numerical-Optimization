import numpy as np

def qp(x, hess_flag=False):
    """
    Quadratic programming example
    """
    f = x[0] ** 2 + x[1] ** 2 + (x[2] + 1) ** 2 # Objective function
    g = np.array([2 * x[0], 2 * x[1], 2 * x[2] + 2]).transpose() # Gradient

    if hess_flag:
        h = np.array([[2, 0, 0], 
                      [0, 2, 0], 
                      [0, 0, 2]])  # Hessian
        return f, g, h
    else:
        return f, g

def qp_eq_constraint():
    """
    Equality constraint for the quadratic programming example
    returns: left_side, right_side
    """
    left_side = np.array([1,1,1]).reshape(1, -1) # left side of the equality constraint as a row vector
    right_side = np.array([1]).reshape(1,-1) # right side of the equality constraint as a row vector
    return left_side, right_side

    
    
def qp_ineq_constraint_1(x, hess_flag=False):
    """
    First inequality constraint for the quadratic programming example
    """
    f = -x[0] # Constraint function
    g = np.array([-1, 0, 0]).transpose() # Gradient
    if hess_flag:
        h = np.array([[0, 0, 0], 
                      [0, 0, 0], 
                      [0, 0, 0]]) # Hessian
        return f, g, h
    else:
        return f, g


def qp_ineq_constraint_2(x, hess_flag=False):
    """
    Second inequality constraint for the quadratic programming example
    """
    f = -x[1] # Constraint function
    g = np.array([0, -1, 0]).transpose() # Gradient
    if hess_flag:
        h = np.array([[0, 0, 0], 
                      [0, 0, 0], 
                      [0, 0, 0]]) # Hessian
        return f, g, h
    else:
        return f, g


def qp_ineq_constraint_3(x, hess_flag=False):
    """
    Third inequality constraint for the quadratic programming example
    """
    f = -x[2] # Constraint function
    g = np.array([0, 0, -1]).transpose() # Gradient
    if hess_flag:
        h = np.array([[0, 0, 0], 
                      [0, 0, 0], 
                      [0, 0, 0]]) # Hessian
        return f, g, h
    else:
        return f, g



def lp(x, hess_flag=False):
    """
    Linear programming example
    """
    f = -x[0] - x[1] # Objective function
    g = np.array([-1, -1]).transpose() # Gradient
    if hess_flag:
        h = np.array([[0, 0], 
                      [0, 0]]) # Hessian
        return f, g, h
    else:
        return f, g


def lp_ineq_constraint_1(x, hess_flag=False):
    """
    First inequality constraint for the linear programming example
    """
    f = -x[0] - x[1] + 1 # Constraint function
    g = np.array([-1, -1]).transpose() # Gradient
    if hess_flag:
        h = np.array([[0, 0], 
                      [0, 0]]) # Hessian
        return f, g, h
    else:
        return f, g


def lp_ineq_constraint_2(x, hess_flag=False):
    """
    Second inequality constraint for the linear programming example
    """
    f = x[1] - 1 # Constraint function
    g = np.array([0, 1]).transpose() # Gradient
    if hess_flag:
        h = np.array([[0, 0], 
                      [0, 0]]) # Hessian
        return f, g, h
    else:
        return f, g


def lp_ineq_constraint_3(x, hess_flag=False):
    """
    Third inequality constraint for the linear programming example
    """
    f = x[0] - 2 # Constraint function
    g = np.array([1, 0]).transpose() # Gradient
    if hess_flag:
        h = np.array([[0, 0], 
                      [0, 0]]) # Hessian
        return f, g, h
    else:
        return f, g


def lp_ineq_constraint_4(x, hess_flag=False):
    """
    Fourth inequality constraint for the linear programming example
    """
    f = -x[1] # Constraint function
    g = np.array([0, -1]).transpose() # Gradient
    if hess_flag:
        h = np.array([[0, 0], 
                      [0, 0]]) # Hessian
        return f, g, h
    else:
        return f, g