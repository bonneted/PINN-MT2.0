"""
plot_util.py
------------
This module contains utility functions for initializing and updating
plots (metrics, parameter evolution, field plots, residuals, etc.) used
in the inverse problem analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import matplotlib.colors as colors
from itertools import chain
from scipy.interpolate import RegularGridInterpolator

def init_metrics(ax, steps, metrics, metrics_names, step_type="iteration", time_unit="s", metrics_idx=[0]):
    """
    Initialize a metric plot on the given axis.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        steps (array-like): x-axis data (iterations or time).
        metrics (list of arrays): List of metric arrays.
        metrics_names (list of str): Labels for each metric.
        step_type (str): 'iteration' or 'time'.
        time_unit (str): Unit label (if step_type is "time").
        metrics_idx (list of int): Indices of the metrics to plot.
    
    Returns:
        lines (list): Line objects (for updating).
        scatters (list): Scatter objects (for highlighting current point).
    """
    ax.set_yscale('log')
    lines = []
    scatters = []
    colors_list = plt.cm.tab10.colors
    for idx in metrics_idx:
        # Plot background (faded) curve
        ax.plot(steps, metrics[idx], alpha=0.2, color=colors_list[idx])
        line, = ax.plot([], [], zorder=3, color=colors_list[idx], label=metrics_names[idx])
        lines.append(line)
        scatters.append(ax.scatter([], [], c='k', zorder=4))
    ax.legend(handlelength=1)
    ax.set_xlabel(f"Time ({time_unit})" if step_type=="time" else "Iterations")
    return lines, scatters

def update_metrics(iteration, lines, scatters, steps, metrics, metrics_idx=[0]):
    """
    Update metric plot lines and scatter positions for the given iteration.
    
    Parameters:
        iteration (int): Current index.
        lines (list): Line objects.
        scatters (list): Scatter objects.
        steps (array-like): x-axis data.
        metrics (list of arrays): List of metric arrays.
        metrics_idx (list of int): Indices of metrics to update.
    
    Returns:
        Updated lines and scatters.
    """
    for line, scatter, idx in zip(lines, scatters, metrics_idx):
        line.set_data(steps[:iteration+1], metrics[idx][:iteration+1])
        scatter.set_offsets([steps[iteration], metrics[idx][iteration]])
    return lines, scatters

def init_variables(axs, steps, param1_actual, param2_actual, param1_history, param2_history, param1_name, param2_name):
    """
    Initialize parameter evolution plots on two given axes.
    
    Parameters:
        axs (list): Two axes for the two parameters.
        steps (array-like): x-axis data.
        param1_actual (float): Reference value for parameter 1.
        param2_actual (float): Reference value for parameter 2.
        param1_history (array-like): History of parameter 1 estimates.
        param2_history (array-like): History of parameter 2 estimates.
        param1_name (str): Label for parameter 1.
        param2_name (str): Label for parameter 2.
    
    Returns:
        line_param1, line_param2, scatter_param1, scatter_param2, axs
    """
    axs[0].hlines(y=param1_actual, xmin=0, xmax=np.max(steps), linestyles='-.', colors="b")
    axs[1].hlines(y=param2_actual, xmin=0, xmax=np.max(steps), linestyles='-.', colors="r")
    line_param1, = axs[0].plot([], [], color='b', zorder=3)
    line_param2, = axs[1].plot([], [], color='r', zorder=3)
    axs[0].plot(steps, param1_history, color='b', alpha=0.2)
    axs[1].plot(steps, param2_history, color='r', alpha=0.2)
    scatter_param1 = axs[0].scatter([], [], c='k', zorder=4,
                                 label=f"{param2_name} = {param2_history[0]:.3f}")#|{param2_actual:.3f}")
    scatter_param2 = axs[1].scatter([], [], c='k', zorder=4,
                                 label=f"{param2_name} = {param2_history[0]:.3f}")#|{param2_actual:.3f}")
    axs[0].legend()
    axs[1].legend()
    return line_param1, line_param2, scatter_param1, scatter_param2, axs

def update_variables(iteration, line_param1, line_param2, scatter_param1, scatter_param2, axs,
                     steps, param1_history, param2_history, param1_actual, param2_actual, param1_name, param2_name):
    """
    Update the parameter evolution plots for the given iteration.
    
    Parameters:
        iteration (int): Current index.
        line_param1, line_param2: Line objects for parameters.
        scatter_param1, scatter_param2: Scatter objects.
        axs (list): Two axes.
        steps (array-like): x-axis data.
        param1_history (array-like): History of parameter 1.
        param2_history (array-like): History of parameter 2.
        param1_actual (float): Reference value for parameter 1.
        param2_actual (float): Reference value for parameter 2.
        param1_name (str): Label for parameter 1.
        param2_name (str): Label for parameter 2.
    
    Returns:
        Updated line and scatter objects and axes.
    """
    line_param1.set_data(steps[:iteration+1], param1_history[:iteration+1])
    scatter_param1.set_offsets([steps[iteration], param1_history[iteration]])
    scatter_param1.set_label(f"{param1_name} = {param1_history[iteration]:.1f}")# | {param1_actual:.1f}")
    
    line_param2.set_data(steps[:iteration+1], param2_history[:iteration+1])
    scatter_param2.set_offsets([steps[iteration], param2_history[iteration]])
    scatter_param2.set_label(f"{param2_name} = {param2_history[iteration]:.3f}")# | {param2_actual:.3f}")

    
    axs[0].legend()
    axs[1].legend()
    return line_param1, line_param2, scatter_param1, scatter_param2, axs


def pcolor_plot(ax, X, Y, C, title, colormap="viridis", norm=None):
    """
    Create a pseudocolor plot on the given axis.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axis to plot on.
        X, Y (2D arrays): Meshgrid coordinates.
        C (2D array): Data for color mapping.
        title (str): Plot title.
        colormap: Colormap to use.
        norm: Normalization for color scaling.
    
    Returns:
        im: The image object from pcolor.
    """
    im = ax.pcolor(X, Y, C, cmap=colormap, shading='auto', norm=norm)
    ax.axis("equal")
    ax.axis("off")
    ax.set_title(title)
    return im

def plot_field(i, fields, field_id, Xmesh, Ymesh, func, ax, field_names, plot_exact=False, colormap="viridis"):
    """
    Plot a computed field (or its exact solution) on the given axis.
    
    Parameters:
        i (int): Time (or iteration) index.
        fields (list): List of field data arrays.
        field_id (int): Field identifier.
        Xmesh, Ymesh (2D arrays): Meshgrid coordinates.
        func (callable): Function returning the exact solution.
        ax (matplotlib.axes.Axes): Axis to plot on.
        field_names (list): List of field names.
        plot_exact (bool): If True, plot the exact solution.
    
    Returns:
        im: The image object from the pcolor plot.
    """
    field = np.array(fields[field_id][i]).reshape(Xmesh.shape)
    field_exact = func(np.hstack((Xmesh.reshape(-1, 1), Ymesh.reshape(-1, 1))))[:, field_id].reshape(Xmesh.shape)
    field_norm = colors.Normalize(vmin=field_exact.min(), vmax=field_exact.max())
    title = f"{field_names[field_id]}*" if plot_exact else field_names[field_id]
    im = pcolor_plot(ax, Xmesh, Ymesh, field_exact if plot_exact else field, title, colormap=colormap, norm=field_norm)
    return im

def plot_field_residual(i, fields, field_id, Xmesh, Ymesh, func, ax, field_names, colormap="coolwarm", ngrid=100):
    """
    Plot the residual (difference between computed and exact field) on the given axis.
    
    Parameters:
        i (int): Time (or iteration) index.
        fields (list): List of field data arrays.
        field_id (int): Field identifier.
        Xmesh, Ymesh (2D arrays): Meshgrid coordinates.
        func (callable): Function returning the exact solution.
        ax (matplotlib.axes.Axes): Axis to plot on.
        colormap (str): Colormap to use for residual.
        ngrid (int): Grid resolution (if needed).
    
    Returns:
        im: The image object from the pcolor plot.
    """
    field_exact = func(np.hstack((Xmesh.reshape(-1, 1), Ymesh.reshape(-1, 1))))[:, field_id].reshape(Xmesh.shape)
    field_diff = np.array(fields[field_id][i]).reshape(Xmesh.shape) - field_exact
    norm = colors.Normalize(vmin=field_diff.min(), vmax=field_diff.max())
    title = f"{field_names[field_id]} - {field_names[field_id]}*"
    im = pcolor_plot(ax, Xmesh, Ymesh, field_diff, title, colormap=colormap, norm=norm)
    return im

def set_normdiff(i, fields, fields_id, func, Xmesh, Ymesh, ngrid=100):
    """
    Compute a normalization object for the residuals across multiple fields.
    
    Parameters:
        i (int): Time (or iteration) index.
        fields (list): List of field data arrays.
        fields_id (list): List of field identifiers.
        func (callable): Function returning the exact solution.
        Xmesh, Ymesh (2D arrays): Meshgrid coordinates.
        ngrid (int): Grid resolution.
    
    Returns:
        normdiff: A Normalize object (for consistent color scaling).
    """
    fields_exact = func(np.hstack((Xmesh.reshape(-1, 1), Ymesh.reshape(-1, 1))))
    cmaxs = []  
    cmins = []
    for fid in fields_id:
        diff = np.array(fields[fid][i]).reshape(ngrid, ngrid) - fields_exact[:, fid].reshape(ngrid, ngrid)
        abs_diff = np.abs(diff)
        cmaxs.append(abs_diff.max())
        cmins.append(-abs_diff.max())
    normdiff = colors.Normalize(vmin=min(cmins), vmax=max(cmaxs))
    return normdiff

def make_formatter():
    """
    Create and return a scalar formatter for colorbar tick labels.
    """
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 3))
    return formatter

def subsample_steps(steps, sub_factors):
    """
    Subsample step indices based on provided subsampling factors.
    
    Parameters:
        steps (list or array): Full list of step indices.
        sub_factors (list): List of subsampling factors for different sections.
    
    Returns:
        Generator of subsampled step indices.
    """
    section_ends = [int(len(steps) * i / len(sub_factors)) for i in range(1, len(sub_factors) + 1)]
    frame_steps_lists = []
    for i, factor in enumerate(sub_factors):
        start = section_ends[i - 1] if i > 0 else 0
        end = section_ends[i]
        frame_steps_lists.append(range(start, end, factor))
    return chain.from_iterable(frame_steps_lists)

def plot_field_columns(fig, ax, fields, fields_id, iteration, Xmesh, Ymesh, func, field_names, colormap, make_formatter):
    """
    Plot exact, computed, and residual fields (with colorbars) for each field.

    Parameters:
      - fig: the matplotlib figure
      - ax: 2D array of axes with shape (3, 1+n_fields); first column reserved for variable/metric plots.
      - fields: a list (or dict) of field data
      - fields_id: list of field ids to plot
      - iteration: current iteration index
      - Xmesh, Ymesh: meshgrid arrays for plotting
      - func: function returning the exact field (used to compute field differences)
      - field_names: list of field names (used for titles)
      - colormap: colormap to use for plotting
      - make_formatter: function to create a scalar formatter for colorbars

    Returns:
      A tuple (ims_ref, ims_field, ims_res) containing lists of image objects.
    """
    ims_ref = []    # List for exact solution images (top row)
    ims_field = []  # List for computed field images (middle row)
    ims_res = []    # List for residual images (bottom row)

    for i, field_id in enumerate(fields_id):
        # --- Exact solution plot (top row) ---
        ax_exact = ax[0][i]
        im_ref = plot_field(iteration, fields, field_id, Xmesh, Ymesh, func, ax_exact,
                            field_names, plot_exact=True, colormap=colormap)
        ims_ref.append(im_ref)
        
        # Manually add a colorbar for the exact field
        pos = ax_exact.get_position()
        cax_pos = mtransforms.Bbox.from_bounds(pos.x0 + pos.width*0.05, pos.y1 + 0.04,
                                               pos.width*0.9, 0.01)
        cax = fig.add_axes(cax_pos)
        cbfield = fig.colorbar(im_ref, cax=cax, orientation='horizontal', format=make_formatter())
        cbfield.ax.xaxis.set_ticks_position('top')
        
        # --- Computed field plot (middle row) ---
        ax_field = ax[1][i]
        im_field = plot_field(iteration, fields, field_id, Xmesh, Ymesh, func, ax_field,
                              field_names, plot_exact=False, colormap=colormap)
        ims_field.append(im_field)
        
        # --- Residual plot (bottom row) ---
        ax_res = ax[2][i]
        im_res = plot_field_residual(iteration, fields, field_id, Xmesh, Ymesh, func, ax_res, field_names)
        ims_res.append(im_res)
        pos = ax_res.get_position()
        cax_pos = mtransforms.Bbox.from_bounds(pos.x0 + pos.width*0.05, pos.y0 - 0.03,
                                               pos.width*0.9, 0.01)
        cax = fig.add_axes(cax_pos)
        fig.colorbar(im_res, cax=cax, orientation='horizontal', format=make_formatter())
    
    return ims_ref, ims_field, ims_res

def animate_frame(i, ims_field, ims_res, line_param1, line_param2, scatter_param1, scatter_param2, ax_variables,
                  lines, scatters, steps, time_unit, step_type, param1_history, param2_history, 
                  param1_actual, param2_actual, param1_name, param2_name, metrics, fields, func, Xmesh, Ymesh, field_names, fields_id):
    """
    Update function for animation frames. This updates parameter plots, metric plots,
    and field (with residual) plots for the given frame index.
    
    Parameters:
        i (int): Frame index.
        ims_field (list): List of image objects for computed fields.
        ims_res (list): List of image objects for residual plots.
        line_param1, line_param2: Line objects for parameter evolution.
        scatter_param1, scatter_param2: Scatter objects for parameters.
        ax_variables (list): Two axes for parameter plots.
        lines, scatters: Objects for metric plots.
        steps (array-like): x-axis data.
        time_unit (str): Time unit (if applicable).
        step_type (str): 'time' or 'iteration'.
        param1_history, param2_history: Parameter histories.
        param1_actual, param2_actual: Reference values.
        param1_name, param2_name: Parameter labels.
        metrics (list): List of metric arrays.
        fields (list): List of field data arrays.
        func (callable): Function returning the exact solution.
        Xmesh, Ymesh (2D arrays): Meshgrid coordinates.
        field_names (list): List of field names.
        fields_id (list): List of field identifiers.
    
    Returns:
        Updated plot objects.
    """
    plt.suptitle(f"{step_type}: {steps[i]:.0f}{f' {time_unit}' if step_type=='time' else ''}", fontsize=16)
    
    # Update parameter evolution plots
    line_param1, line_param2, scatter_param1, scatter_param2, ax_variables = update_variables(
        i, line_param1, line_param2, scatter_param1, scatter_param2, ax_variables,
        steps, param1_history, param2_history, param1_actual, param2_actual, param1_name, param2_name)
    
    # Update metric plots
    lines, scatters = update_metrics(i, lines, scatters, steps, metrics)
    
    # Update field and residual plots for each field
    for idx, fid in enumerate(fields_id):
        field = np.array(fields[fid][i]).reshape(Xmesh.shape)
        field_exact = func(np.hstack((Xmesh.reshape(-1, 1), Ymesh.reshape(-1, 1))))[:, fid].reshape(Xmesh.shape)
        field_diff = field - field_exact
        
        ims_field[idx].set_array(field.ravel())
        ims_res[idx].set_array(field_diff.ravel())
        norm = colors.Normalize(vmin=field_diff.min(), vmax=field_diff.max())
        ims_res[idx].set_norm(norm)
    
    return ims_field, ims_res, line_param1, line_param2, scatter_param1, scatter_param2, ax_variables, lines, scatters

def moving_average(arr, window_size):
    window = np.ones(window_size) / window_size
    return np.apply_along_axis(lambda x: np.convolve(x, window, mode='valid'), 0, arr)

def moving_min(arr, window_size):
    new_arr = np.zeros((arr.shape[0] - window_size + 1, arr.shape[1]))
    for n_metric in range(arr.shape[1]):
        for i in range(new_arr.shape[0]):
            new_arr[i, n_metric] = np.min(arr[i:i+window_size, n_metric])
    return new_arr