import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_function_values(result_newton_path=None, function_name=''):
    """
    Function to plot the function values at each iteration for given methods.
    :param result_newton_path: List of tuples (#iteration and objective value) taken by constrained_min using Newton's method
    :param function_name: Name of the function 
    """
    # iterate over the results and print the results
    for iteration, f_value, _ in result_newton_path:
        print(f'iteration: {iteration}, f_value: {f_value}')

    if result_newton_path:
        # plt.plot(result_newton_path, label='Newton')
        plt.plot([iteration for iteration, f_value, _ in result_newton_path], [f_value for iteration, f_value, _ in result_newton_path], label='Newton')
        # add a point and the f_value of every iteration on the plot (bold text) and font calibri
        for iteration, f_value, _ in result_newton_path:
            plt.text(iteration, f_value, f'{f_value:.4f}', fontsize=9, color='black', fontweight='bold', fontname='Arial') 
            # add a point on the plot
            plt.scatter(iteration, f_value, color='red', s=10)



    # Set the x-axis label
    plt.xlabel('Iteration')

    # Set the y-axis label
    plt.ylabel('Function Value')
    
    # Set the title
    plt.title(f'Function Values vs. Iteration Number,  {function_name}')

    # Show the plot
    plt.show()





def qp_plot_3d(path, title):
    """
    Plot the path of the optimization in 3D for the quadratic programming example.
    :param path (list of tuples): Path history. Each tuple contains the iteration number, function value and x values
    :param title: Title of the plot
    """
    ax = plt.figure(figsize=(15, 9)).add_subplot(projection='3d') # Create a 3D plot
    ax.set_xlabel('X axis') # Set the x-axis label
    ax.set_ylabel('Y axis') # Set the y-axis label
    ax.set_zlabel('Z axis') # Set the z-axis label
    ax.set_title(f"{title} path", fontsize=20, fontweight='bold') # Set the title of the plot
    # assign the boundaries of the axis
    ax.set_xlim(0, 1) # Set the x-axis limits
    ax.set_ylim(0, 1) # Set the y-axis limits
    ax.set_zlim(0, 1) # Set the z-axis limits
    # add grid to the plot
    ax.grid(True)
    x_limit = np.array([1, 0, 0]) # Define the x-axis limits
    y_limit = np.array([0, 1, 0]) # Define the y-axis limits
    z_limit = np.array([0, 0, 1]) # Define the z-axis limits
    limits = [x_limit, y_limit, z_limit] # Define the limits of the plot
    shape = Poly3DCollection([limits], alpha=.25, color='b') # Create a 3D shape
    plt.gca().add_collection3d(shape) # Add the shape to the plot. 

    #  path is lsit of tuple, each tuple contains the iteration number, function value and x values. get the x values from the path 
    x_y_z_values = np.array([x for _, _, x in path]) # Get the x, y, z values from the path history

    x = x_y_z_values[:, 0] # Get the x values
    y = x_y_z_values[:, 1] # Get the y values
    z = x_y_z_values[:, 2] # Get the z values

    plt.plot(x[:-1], y[:-1], z[:-1], '-o', label='path') # Plot the path
    plt.plot(x[-1], y[-1], z[-1], marker='^', markerfacecolor='yellow', markersize=12, label='final x') # Plot the final x
 
    plt.show() # Show the plot


def function(x):
    return 1 - x

def lp_plot_2d(path, title):
    """
    Plot the path of the optimization in 2D for the linear programming example.
    :param path_history: Path history. Each row contains the x values, function value and the path.
    :param title: Title of the plot
    """
    # x0_len = len(x0) # Get the length of the initial point

    x_y_values = np.array([x for _, _, x in path]) # Get the x, y, z values from the path history# Get the x, y values, function value and path from the path history
    x = x_y_values[:, 0] # Get the x values
    y = x_y_values[:, 1] # Get the y values
    # y >= 0
    x1_lim = np.linspace(function(0), 2, 50) # Define the x values for the first limit
    y1_lim = np.zeros(50)  # Define the y values for the first limit
    # y <=1
    x2_lim = np.linspace(function(1), 2, 50) # Define the x values for the second limit
    y2_lim = np.ones(50) * 1 # Define the y values for the second limit
    # x <=2
    x3_lim = np.ones(50) * 2 # Define the x values for the third limit
    y3_lim = np.linspace(0, 1, 50) # Define the y values for the third limit
    # 0 >= 1-x-y
    x4_lim = function(y3_lim) # Define the x values for the fourth limit
    y4_lim = np.linspace(0, 1, 50) # Define the y values for the fourth limit
    fig, ax = plt.subplots(figsize=(15, 10)) # Create a plot 

    # assign the boundaries of the axis
    ax.set_xlim(-0.5, 2.5) # Set the x-axis limits
    ax.set_ylim(-0.5, 1.5) # Set the y-axis limits
    
    ax.grid(True) # Add grid to the plot

    ax.plot(x[:-1], y[:-1], '-o', label='path', color='r') # Plot the path in red. The path is the x and y values
    ax.plot(x[-1], y[-1], marker='^', markerfacecolor='yellow', markersize=12, label='final x') # Plot the final x in yellow
    ax.plot(x1_lim, y1_lim, color='y') # Plot the first limit in yellow
    ax.plot(x2_lim, y2_lim, color='y') # Plot the second limit in yellow
    ax.plot(x3_lim, y3_lim, color='y') # Plot the third limit in yellow
    ax.plot(x4_lim, y4_lim, color='y') # Plot the fourth limit in yellow
    ax.set_xlabel("X axis") # Set the x-axis label
    ax.set_ylabel("Y axis") # Set the y-axis label
    ax.set_title(f"{title} path", fontsize=20, fontweight='bold') # Set the title of the plot
    # boundaries colored
    plt.fill_between(x4_lim, function(x4_lim), y2_lim, color='yellow', alpha=0.25) # Fill the area between the fourth limit and the second limit in yellow
    plt.fill_between(x1_lim, y1_lim, y2_lim, color='yellow', alpha=0.25) # Fill the area between the first limit and the second limit in yellow
    plt.show()


