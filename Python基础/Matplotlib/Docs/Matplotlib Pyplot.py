# Databricks notebook source
# MAGIC %md
# MAGIC # Matplotlib Pyplot
# MAGIC 
# MAGIC Pyplot 是 Matplotlib 的子库，提供了和 MATLAB 类似的绘图 API。
# MAGIC 
# MAGIC Pyplot 是常用的绘图模块，能很方便让用户绘制 2D 图表。
# MAGIC 
# MAGIC Pyplot 包含一系列绘图函数的相关函数，每个函数会对当前的图像进行一些修改，例如：给图像加上标记，生新的图像，在图像中产生新的绘图区域等等。

# COMMAND ----------

import matplotlib.pyplot as plt
# 或者
from matplotlib import pyplot as plt

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import math

x = np.arange(0, math.pi*2, 0.05)
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 功能
# MAGIC 
# MAGIC https://matplotlib.org/stable/api/pyplot_summary.html
# MAGIC 
# MAGIC | 功能 | 描述 |
# MAGIC | ----------- | ----------- |
# MAGIC | acorr(x, *[, data]) | Plot the autocorrelation of x. |
# MAGIC | angle_spectrum(x[, Fs, Fc, window, pad_to, ...]) | Plot the angle spectrum. |
# MAGIC | annotate(text, xy, *args, **kwargs) | Annotate the point xy with text text. |
# MAGIC | arrow(x, y, dx, dy, **kwargs) | Add an arrow to the Axes. |
# MAGIC | autoscale([enable, axis, tight]) | Autoscale the axis view to the data (toggle). |
# MAGIC | autumn() | Set the colormap to 'autumn'. |
# MAGIC | axes([arg]) | Add an axes to the current figure and make it the current axes. |
# MAGIC | axhline([y, xmin, xmax]) | Add a horizontal line across the axis. |
# MAGIC | axhspan(ymin, ymax[, xmin, xmax]) | Add a horizontal span (rectangle) across the Axes. |
# MAGIC | axis(*args[, emit]) | Convenience method to get or set some axis properties. |
# MAGIC | axline(xy1[, xy2, slope]) | Add an infinitely long straight line. |
# MAGIC | axvline([x, ymin, ymax]) | Add a vertical line across the Axes. |
# MAGIC | axvspan(xmin, xmax[, ymin, ymax]) | Add a vertical span (rectangle) across the Axes. |
# MAGIC | bar(x, height[, width, bottom, align, data]) | Make a bar plot. |
# MAGIC | bar_label(container[, labels, fmt, ...]) | Label a bar plot. |
# MAGIC | barbs(*args[, data]) | Plot a 2D field of barbs. |
# MAGIC | barh(y, width[, height, left, align]) | Make a horizontal bar plot. |
# MAGIC | bone() | Set the colormap to 'bone'. |
# MAGIC | box([on]) | Turn the axes box on or off on the current axes. |
# MAGIC | boxplot(x[, notch, sym, vert, whis, ...]) | Draw a box and whisker plot. |
# MAGIC | broken_barh(xranges, yrange, *[, data]) | Plot a horizontal sequence of rectangles. |
# MAGIC | cla() | Clear the current axes. |
# MAGIC | clabel(CS[, levels]) | Label a contour plot. |
# MAGIC | clf() | Clear the current figure. |
# MAGIC | clim([vmin, vmax]) | Set the color limits of the current image. |
# MAGIC | close([fig]) | Close a figure window. |
# MAGIC | cohere(x, y[, NFFT, Fs, Fc, detrend, ...]) | Plot the coherence between x and y. |
# MAGIC | colorbar([mappable, cax, ax]) | Add a colorbar to a plot. |
# MAGIC | connect(s, func) | Bind function func to event s. |
# MAGIC | contour(*args[, data]) | Plot contour lines. |
# MAGIC | contourf(*args[, data]) | Plot filled contours. |
# MAGIC | cool() | Set the colormap to 'cool'. |
# MAGIC | copper() | Set the colormap to 'copper'. |
# MAGIC | csd(x, y[, NFFT, Fs, Fc, detrend, window, ...]) | Plot the cross-spectral density. |
# MAGIC | delaxes([ax]) | Remove an Axes (defaulting to the current axes) from its figure. |
# MAGIC | disconnect(cid) | Disconnect the callback with id cid. |
# MAGIC | draw() | Redraw the current figure. |
# MAGIC | draw_if_interactive() | Redraw the current figure if in interactive mode. |
# MAGIC | errorbar(x, y[, yerr, xerr, fmt, ecolor, ...]) | Plot y versus x as lines and/or markers with attached errorbars. |
# MAGIC | eventplot(positions[, orientation, ...]) | Plot identical parallel lines at the given positions. |
# MAGIC | figimage(X[, xo, yo, alpha, norm, cmap, ...]) | Add a non-resampled image to the figure. |
# MAGIC | figlegend(*args, **kwargs) | Place a legend on the figure. |
# MAGIC | fignum_exists(num) | Return whether the figure with the given id exists. |
# MAGIC | figtext(x, y, s[, fontdict]) | Add text to figure. |
# MAGIC | figure([num, figsize, dpi, facecolor, ...]) | Create a new figure, or activate an existing figure. |
# MAGIC | fill(*args[, data]) | Plot filled polygons. |
# MAGIC | fill_between(x, y1[, y2, where, ...]) | Fill the area between two horizontal curves. |
# MAGIC | fill_betweenx(y, x1[, x2, where, step, ...]) | Fill the area between two vertical curves. |
# MAGIC | findobj([o, match, include_self]) | Find artist objects. |
# MAGIC | flag() | Set the colormap to 'flag'. |
# MAGIC | gca(**kwargs) | Get the current Axes. |
# MAGIC | gcf() | Get the current figure. |
# MAGIC | gci() | Get the current colorable artist. |
# MAGIC | get(obj, *args, **kwargs) | Return the value of an Artist's property, or print all of them. |
# MAGIC | get_current_fig_manager() | Return the figure manager of the current figure. |
# MAGIC | get_figlabels() | Return a list of existing figure labels. |
# MAGIC | get_fignums() | Return a list of existing figure numbers. |
# MAGIC | get_plot_commands() | Get a sorted list of all of the plotting commands. |
# MAGIC | getp(obj, *args, **kwargs) | Return the value of an Artist's property, or print all of them. |
# MAGIC | ginput([n, timeout, show_clicks, mouse_add, ...]) | Blocking call to interact with a figure. |
# MAGIC | gray() | Set the colormap to 'gray'. |
# MAGIC | grid([visible, which, axis]) | Configure the grid lines. |
# MAGIC | hexbin(x, y[, C, gridsize, bins, xscale, ...]) | Make a 2D hexagonal binning plot of points x, y. |
# MAGIC | hist(x[, bins, range, density, weights, ...]) | Plot a histogram. |
# MAGIC | hist2d(x, y[, bins, range, density, ...]) | Make a 2D histogram plot. |
# MAGIC | hlines(y, xmin, xmax[, colors, linestyles, ...]) | Plot horizontal lines at each y from xmin to xmax. |
# MAGIC | hot() | Set the colormap to 'hot'. |
# MAGIC | hsv() | Set the colormap to 'hsv'. |
# MAGIC | imread(fname[, format]) | Read an image from a file into an array. |
# MAGIC | imsave(fname, arr, **kwargs) | Save an array as an image file. |
# MAGIC | imshow(X[, cmap, norm, aspect, ...]) | Display data as an image, i.e., on a 2D regular raster. |
# MAGIC | inferno() | Set the colormap to 'inferno'. |
# MAGIC | install_repl_displayhook() | Install a repl display hook so that any stale figure are automatically redrawn when control is returned to the repl. |
# MAGIC | ioff() | Disable interactive mode. |
# MAGIC | ion() | Enable interactive mode. |
# MAGIC | isinteractive() | Return whether plots are updated after every plotting command. |
# MAGIC | jet() | Set the colormap to 'jet'. |
# MAGIC | legend(*args, **kwargs) | Place a legend on the Axes. |
# MAGIC | locator_params([axis, tight]) | Control behavior of major tick locators. |
# MAGIC | loglog(*args, **kwargs) | Make a plot with log scaling on both the x and y axis. |
# MAGIC | magma() | Set the colormap to 'magma'. |
# MAGIC | magnitude_spectrum(x[, Fs, Fc, window, ...]) | Plot the magnitude spectrum. |
# MAGIC | margins(*margins[, x, y, tight]) | Set or retrieve autoscaling margins. |
# MAGIC | matshow(A[, fignum]) | Display an array as a matrix in a new figure window. |
# MAGIC | minorticks_off() | Remove minor ticks from the Axes. |
# MAGIC | minorticks_on() | Display minor ticks on the Axes. |
# MAGIC | new_figure_manager(num, *args, **kwargs) | Create a new figure manager instance. |
# MAGIC | nipy_spectral() | Set the colormap to 'nipy_spectral'. |
# MAGIC | pause(interval) | Run the GUI event loop for interval seconds. |
# MAGIC | pcolor(*args[, shading, alpha, norm, cmap, ...]) | Create a pseudocolor plot with a non-regular rectangular grid. |
# MAGIC | pcolormesh(*args[, alpha, norm, cmap, vmin, ...]) | Create a pseudocolor plot with a non-regular rectangular grid. |
# MAGIC | phase_spectrum(x[, Fs, Fc, window, pad_to, ...]) | Plot the phase spectrum. |
# MAGIC | pie(x[, explode, labels, colors, autopct, ...]) | Plot a pie chart. |
# MAGIC | pink() | Set the colormap to 'pink'. |
# MAGIC | plasma() | Set the colormap to 'plasma'. |
# MAGIC | plot(*args[, scalex, scaley, data]) | Plot y versus x as lines and/or markers. |
# MAGIC | plot_date(x, y[, fmt, tz, xdate, ydate, data]) | Plot coercing the axis to treat floats as dates. |
# MAGIC | polar(*args, **kwargs) | Make a polar plot. |
# MAGIC | prism() | Set the colormap to 'prism'. |
# MAGIC | psd(x[, NFFT, Fs, Fc, detrend, window, ...]) | Plot the power spectral density. |
# MAGIC | quiver(*args[, data]) | Plot a 2D field of arrows. |
# MAGIC | quiverkey(Q, X, Y, U, label, **kwargs) | Add a key to a quiver plot. |
# MAGIC | rc(group, **kwargs) | Set the current rcParams. group is the grouping for the rc, e.g., for the group is , for , the group is , and so on. Group may also be a list or tuple of group names, e.g., (xtick, ytick). kwargs is a dictionary attribute name/value pairs, e.g.,::.lines.linewidthlinesaxes.facecoloraxes |
# MAGIC | rc_context([rc, fname]) | Return a context manager for temporarily changing rcParams. |
# MAGIC | rcdefaults() | Restore the rcParams from Matplotlib's internal default style. |
# MAGIC | rgrids([radii, labels, angle, fmt]) | Get or set the radial gridlines on the current polar plot. |
# MAGIC | savefig(*args, **kwargs) | Save the current figure. |
# MAGIC | sca(ax) | Set the current Axes to ax and the current Figure to the parent of ax. |
# MAGIC | scatter(x, y[, s, c, marker, cmap, norm, ...]) | A scatter plot of y vs. |
# MAGIC | sci(im) | Set the current image. |
# MAGIC | semilogx(*args, **kwargs) | Make a plot with log scaling on the x axis. |
# MAGIC | semilogy(*args, **kwargs) | Make a plot with log scaling on the y axis. |
# MAGIC | set_cmap(cmap) | Set the default colormap, and applies it to the current image if any. |
# MAGIC | set_loglevel(*args, **kwargs) | Set Matplotlib's root logger and root logger handler level, creating the handler if it does not exist yet. |
# MAGIC | setp(obj, *args, **kwargs) | Set one or more properties on an Artist, or list allowed values. |
# MAGIC | show(*[, block]) | Display all open figures. |
# MAGIC | specgram(x[, NFFT, Fs, Fc, detrend, window, ...]) | Plot a spectrogram. |
# MAGIC | spring() | Set the colormap to 'spring'. |
# MAGIC | spy(Z[, precision, marker, markersize, ...]) | Plot the sparsity pattern of a 2D array. |
# MAGIC | stackplot(x, *args[, labels, colors, ...]) | Draw a stacked area plot. |
# MAGIC | stairs(values[, edges, orientation, ...]) | A stepwise constant function as a line with bounding edges or a filled plot. |
# MAGIC | stem(*args[, linefmt, markerfmt, basefmt, ...]) | Create a stem plot. |
# MAGIC | step(x, y, *args[, where, data]) | Make a step plot. |
# MAGIC | streamplot(x, y, u, v[, density, linewidth, ...]) | Draw streamlines of a vector flow. |
# MAGIC | subplot(*args, **kwargs) | Add an Axes to the current figure or retrieve an existing Axes. |
# MAGIC | subplot2grid(shape, loc[, rowspan, colspan, fig]) | Create a subplot at a specific location inside a regular grid. |
# MAGIC | subplot_mosaic(mosaic, *[, sharex, sharey, ...]) | Build a layout of Axes based on ASCII art or nested lists. |
# MAGIC | subplot_tool([targetfig]) | Launch a subplot tool window for a figure. |
# MAGIC | subplots([nrows, ncols, sharex, sharey, ...]) | Create a figure and a set of subplots. |
# MAGIC | subplots_adjust([left, bottom, right, top, ...]) | Adjust the subplot layout parameters. |
# MAGIC | summer() | Set the colormap to 'summer'. |
# MAGIC | suptitle(t, **kwargs) | Add a centered suptitle to the figure. |
# MAGIC | switch_backend(newbackend) | Close all open figures and set the Matplotlib backend. |
# MAGIC | table([cellText, cellColours, cellLoc, ...]) | Add a table to an Axes. |
# MAGIC | text(x, y, s[, fontdict]) | Add text to the Axes. |
# MAGIC | thetagrids([angles, labels, fmt]) | Get or set the theta gridlines on the current polar plot. |
# MAGIC | tick_params([axis]) | Change the appearance of ticks, tick labels, and gridlines. |
# MAGIC | ticklabel_format(*[, axis, style, ...]) | Configure the ScalarFormatter used by default for linear axes. |
# MAGIC | tight_layout(*[, pad, h_pad, w_pad, rect]) | Adjust the padding between and around subplots. |
# MAGIC | title(label[, fontdict, loc, pad, y]) | Set a title for the Axes. |
# MAGIC | tricontour(*args, **kwargs) | Draw contour lines on an unstructured triangular grid. |
# MAGIC | tricontourf(*args, **kwargs) | Draw contour regions on an unstructured triangular grid. |
# MAGIC | tripcolor(*args[, alpha, norm, cmap, vmin, ...]) | Create a pseudocolor plot of an unstructured triangular grid. |
# MAGIC | triplot(*args, **kwargs) | Draw a unstructured triangular grid as lines and/or markers. |
# MAGIC | twinx([ax]) | Make and return a second axes that shares the x-axis. |
# MAGIC | twiny([ax]) | Make and return a second axes that shares the y-axis. |
# MAGIC | uninstall_repl_displayhook() | Uninstall the Matplotlib display hook. |
# MAGIC | violinplot(dataset[, positions, vert, ...]) | Make a violin plot. |
# MAGIC | viridis() | Set the colormap to 'viridis'. |
# MAGIC | vlines(x, ymin, ymax[, colors, linestyles, ...]) | Plot vertical lines at each x from ymin to ymax. |
# MAGIC | waitforbuttonpress([timeout]) | Blocking call to interact with the figure. |
# MAGIC | winter() | Set the colormap to 'winter'. |
# MAGIC | xcorr(x, y[, normed, detrend, usevlines, ...]) | Plot the cross correlation between x and y. |
# MAGIC | xkcd([scale, length, randomness]) | Turn on xkcd sketch-style drawing mode. |
# MAGIC | xlabel(xlabel[, fontdict, labelpad, loc]) | Set the label for the x-axis. |
# MAGIC | xlim(*args, **kwargs) | Get or set the x limits of the current axes. |
# MAGIC | xscale(value, **kwargs) | Set the x-axis scale. |
# MAGIC | xticks([ticks, labels]) | Get or set the current tick locations and labels of the x-axis. |
# MAGIC | ylabel(ylabel[, fontdict, labelpad, loc]) | Set the label for the y-axis. |
# MAGIC | ylim(*args, **kwargs) | Get or set the y-limits of the current axes. |
# MAGIC | yscale(value, **kwargs) | Set the y-axis scale. |
# MAGIC | yticks([ticks, labels]) | Get or set the current tick locations and labels of the y-axis. |
