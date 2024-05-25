import matplotlib.pyplot as plt
import numpy as np

def plot_contour(function, path_gd, path_newton, function_name='', x1_min=-2, x1_max=2, x2_min=-2, x2_max=2, levels=25):
    """
    Function to plot the contour lines of a given function and paths taken by unconstrained_min.
    :param function: Objective function
    :param path_gd (List): List of tuples (#iteration, objective value, x (x1,x2)) taken by unconstrained_min using gradient descent
    :param path_newton (List): List of tuples (#iteration, objective value, x (x1,x2)) taken by unconstrained_min using Newton's method
    :param x1_min: Minimum value of x1
    :param x1_max: Maximum value of x1
    :param x2_min: Minimum value of x2
    :param x2_max: Maximum value of x2
    :param function_name: Name of the function
    :param levels: Levels for contour lines
    """
    # Generate x and y values within the specified limits
    x = np.linspace(x1_min, x1_max, 100)
    y = np.linspace(x2_min, x2_max, 100)
    # Compute the objective function at points from x,y and plot it in the contour plot. the functions gets as  input only arrays of shape (,2), for example functiopn (x[0], y[0])
    
    X, Y = np.meshgrid(x, y) # Create meshgrid of x and y values to evaluate the function on a grid of points
    

    # Create an empty array to store the objective function values
    Z = np.zeros(X.shape)

    # Evaluate the objective function at each point in the grid
    for i in range(len(x)):
        for j in range(len(y)):
            Z[i, j], _ = function(np.array([[X[i,j]],[Y[i,j]]]))
            # print(f'Z[i, j].shape: {Z[i, j].shape}')
    

    # Plot the contour lines
    contour = plt.contour(X, Y, Z, levels=levels)
    
    # Plot the paths and their names in the legend
    if path_gd:
        _, f_Values, x_list = zip(*path_gd)
        # Plot the f_Values on top of the contour plot according to the x1 and x2 values. all list are in the same length, such that f_Values[i] is the value of the function at x1_list[i], x2_list[i].
        x1_list = [x[0] for x in x_list]
        x2_list = [x[1] for x in x_list]
        plt.plot(x1_list, x2_list,  marker='o', color='blue', linestyle='dashed', label='Gradient Descent')
        # mark the end point (last value of x1_list and x2_list) with a blue x that is bigger and with edge color black
        plt.plot(x1_list[-1], x2_list[-1], marker='X', markersize=10, color='blue', markeredgewidth=2, markeredgecolor='black')
 

    if path_newton:
        _, f_Values, x_list = zip(*path_newton)
        x1_list = [x[0] for x in x_list]
        x2_list = [x[1] for x in x_list]
        # Plot the f_Values on top of the contour plot
        plt.plot(x1_list, x2_list, marker='o',  color='orange', linestyle='dashed', label='Newton')
        # mark the end point (last value of x1_list and x2_list) with a orange X that is bigger and with edge color black
        plt.plot(x1_list[-1], x2_list[-1], marker='X', markersize=10, color='orange', markeredgewidth=2, markeredgecolor='black')


    # Add labels to the contour lines
    plt.clabel(contour, inline=True, fontsize=8)
    plt.legend()
    # add labels to x1 and x2
    plt.xlabel('x1')
    plt.ylabel('x2')

    # Set the title with the name of the function
    plt.title(f'Contour Plot of Objective Function {function_name}')

    # Show the plot
    plt.show()


# c. A utility that plots function values at each iteration, for given methods (on the same, single plots) to enable comparison of the decrease in function values of methods.
def plot_function_values(result_gd_path=None, result_newton_path=None, function_name='',):
    """
    Function to plot the function values at each iteration for given methods.
    :param result_gd_path: List of tuples (#iteration and objective value) taken by unconstrained_min using gradient descent
    :param result_newton_path: List of tuples (#iteration and objective value) taken by unconstrained_min using Newton's method
    """

    # if both of them are none
    if result_gd_path is None and result_newton_path is None:
        raise ValueError("Both result_gd_path and result_newton_path cannot be None")
    
    # Plot the function values at each iteration for gradient descent and Newton's method
    if result_gd_path:
        # resuld_gd_path is a list of tuples (#iteration, objective value). plot the objective value vs iteration number 
        plt.plot([iteration for iteration, f_value, _ in result_gd_path], [f_value for iteration, f_value , _ in result_gd_path], label='Gradient Descent')

        # plt.plot(result_gd_path, label='Gradient Descent')

    if result_newton_path:
        # plt.plot(result_newton_path, label='Newton')
        plt.plot([iteration for iteration, f_value, _ in result_newton_path], [f_value for iteration, f_value, _ in result_newton_path], label='Newton')


    # Set the x-axis label
    plt.xlabel('Iteration')

    # Set the y-axis label
    plt.ylabel('Function Value')
    

    # Set the title
    plt.title(f'Function Values vs. Iteration Number, Function {function_name}')
    
    # Show the legend
    plt.legend()
    
    # Show the plot
    plt.show()