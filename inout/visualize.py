# Plot holograms and images
import sys
sys.path.insert(0, '..')

from image import HPOEImage
import matplotlib.pyplot as plt
import numpy as np
import copy

def plot_image(image, axis_units = 'pixel', scaling='auto', title = None, mode='intensity'):
    """
    Plots the image provided in the "image" parameter.

    Parameters
    ----------
    image : HPOEImage object or numpy array
        image to be plotted
    axis_units: string (optional)
        selects the units for the x and y axis. Either 'pixel' or 'space'
    scaling: string or two element list (optional)
        if 'auto', it scales the image so that the minimum value of the image corresponds to 0 and
        the max to 1. 
        if a two element list, it will scale the image so that scaling[0] --> 0, scaling[1] --> 1.
        If None, no scaling is performed.
    title: string (optional)
        optional title for the plot
    mode: string
        Indicates what to plot. Can be one of 'intensity' (plots only intensity), 'phase' (only phase)
        or 'intensity_and_phase' (plots both)
    """

    ################################
    # Check the image and make some operations if necessary
    if isinstance(image, HPOEImage):
        im_cnt = image.im
        im_content = copy.deepcopy(im_cnt)
    else:
        im_content = image
        
    ################################
    # Formatting stuff
    if axis_units not in ['pixel', 'space']:
        print('The plot axis units are not recognized. Defaulting to pixels.')
        axis_units = 'pixel'

    fig = plt.figure()
    plt.gray()

    ratio = 1
    spacing = [1, 1]

    if isinstance(image, HPOEImage):
        ratio = image.pixel_spacing[0]/image.pixel_spacing[1]
        spacing = image.pixel_spacing

    if mode == 'intensity' or mode == 'phase':

        # We need only one plot
        ax = fig.add_subplot(111)
        single_plot(fig, ax, im_content, axis_units, title, mode, spacing, ratio, scaling)

    elif mode == 'intensity_and_phase':

        # We need two plots

        # First subplot is intensity
        ax = fig.add_subplot(121)
        single_plot(fig, ax, im_content, axis_units, title, 'intensity', spacing, ratio, scaling)

        # Second subplot is phase
        ax2 = fig.add_subplot(122, sharex=ax, sharey=ax)
        single_plot(fig, ax2, im_content, axis_units, title, 'phase', spacing, ratio, scaling)

    plt.draw()
    plt.pause(0.001)

    return fig, ax

def single_plot(fig, ax, im_cont, axis_units, title, mode, spacing, ratio, scaling):

    im_content = copy.deepcopy(im_cont)

    ax.set_xlabel('x (%s)' % axis_units)
    ax.set_ylabel('y (%s)' % axis_units)
    
    if title is not None:
        ax.set_title('%s, %s' % (title, mode))

    if mode == 'intensity':
        im_content = np.abs(im_content)
    elif mode == 'phase':
        im_content = np.angle(im_content)

    if scaling == 'auto':
        scaling = [np.amin(im_content), np.amax(im_content)]

    if scaling is not None:
        im_content = np.maximum(im_content, scaling[0])
        im_content = np.minimum(im_content, scaling[1])
        #im_content = (im_content-scaling[0])/(scaling[1]-scaling[0])
        v_min = scaling[0]
        v_max = scaling[1]
    else:
        v_min = np.amin(im_content)
        v_max =np.amax(im_content)
    
    ################################
    # Actual plot
    if len(im_content.shape) > 2:
        if im_content.shape[2] == 1:
            # Choose first channel
            plt_content = im_content[:,:,0]   
    else:
        plt_content = im_content

    implt = ax.imshow(plt_content,  origin='upper', interpolation='antialiased', aspect=ratio, vmin=v_min, vmax=v_max)

    # We use colorbar only if grayscale
    if len(im_content.shape) > 2:
        if im_content.shape[2] == 1:
            fig.colorbar(implt)
    else:
        fig.colorbar(implt)
    
    if axis_units == 'space':
        # yticks use elemnt 0 and xticks element 1 because matplotlib uses different convention
        plt.yticks(range(0, im_content.shape[0], 200), np.arange(0, im_content.shape[0], 200)*spacing[0])
        plt.xticks(range(0, im_content.shape[1], 200), np.arange(0, im_content.shape[1], 200)*spacing[1])
    
    
class ImageForCoordExtraction(object):
    """ 
    Class for showing an image and extracting the coordinate of the first click the user makes.
    """

    def __init__(self):

        self.point = ()

    def get_coord(self, image, show_lines=True):

        self.fig, self.ax = plot_image(image, axis_units='pixel', scaling='auto', title = 'Click on desired coordinate')

        if show_lines:
            self.lx = self.ax.axhline(color='r')  # the horiz line
            self.ly = self.ax.axvline(color='r')  # the vert line
            cid = self.fig.canvas.mpl_connect('button_press_event', self.__on_click__)

        cid2 = self.fig.canvas.mpl_connect('motion_notify_event', self.__mouse_move__)
        plt.show()
        return self.point

    def __on_click__(self,click):
        self.point = (click.xdata,click.ydata)
        plt.close()
        return self.point

    def __mouse_move__(self,event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata

        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        self.ax.figure.canvas.draw()


