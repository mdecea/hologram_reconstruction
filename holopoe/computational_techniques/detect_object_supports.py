# Attempts to detect object supports in a given image, to use in iterative reconstruction processes

from holopoe.image import HPOEImage
import copy
from skimage.filters import try_all_threshold, threshold_mean, threshold_isodata, threshold_local
from skimage.filters import threshold_li, threshold_minimum, threshold_otsu, threshold_yen, threshold_triangle
from skimage.morphology import binary_dilation
from holopoe.process.image_processing import convert_to_grayscale
import numpy as np
import matplotlib.pyplot as plt

def test_thresholds(image):
    """
    Creates a figure with all the possible algorithms for thresholding that are implemented in skimage
    Use this to find out which is the best method for thresholding for a specific hologram.

    Parameters
    ----------
    image : HPOEImage or numpy array
       The image where we want to find the object supports.
    """

    # ---------- PRE-PROCESS -------------

    # If the provided image is an HPOEImage, extract the image matrix
    if isinstance(image, HPOEImage):
        im_cnt = image.im
        im_content = copy.deepcopy(im_cnt)
    else:
        im_content = image

    # Make sure ft has only 1 element in z, and if not convert to grayscale image
    if im_content.shape[2] != 1:
        # Convert to grayscale
        print('The provided field is RGB. Converting to grayscale for propagation.')
        field = convert_to_grayscale(im_content, method = 'weighted')
    
    # Remove 3rd dimension
    im_content = np.squeeze(im_content)

    # If the data is complex, get the absolute value (we do thresholding on the intensity)
    im_content = np.abs(im_content)

    # ----------- THRESHOLDING ------------
    fig, ax = try_all_threshold(im_content, figsize=(10, 8), verbose=False)
    plt.show()

def get_obj_supp_threshold(image, method='minimum', obj_color='dark', pad=0, plot_process=False):

    """
    Detects the object supports by thesholding. Returns a mask with '0' where there is an object.
    
    Parameters
    ----------
    image : HPOEImage or numpy array
       The image where we want to find the object supports.
    method: string. Either 'isodata', 'li', 'mean', 'minimum', 'otsu', 'triangle', 'yen', 'adaptive', 'local'
        The thresholding method to use. It is likely that different images require a different method.
        From VERY LIMITED testing, minimum and yen seem to be the best.
    obj_color: string, either 'dark' or 'light'
       Indicates if the objects are dark in a lighter background (obj_color = 'dark') or viceversa (obj_color = 'light')
    pad: int
        The detected supports are expanded by the amount in pad.
    plot_process: boolean
        If True, it plots the thresholding process. Mostly for debugging.
    """

    # ---------- PRE-PROCESS -------------

    # If the provided image is an HPOEImage, extract the image matrix
    if isinstance(image, HPOEImage):
        im_cnt = image.im
        im_content = copy.deepcopy(im_cnt)
    else:
        im_content = image

    # Make sure ft has only 1 element in z, and if not convert to grayscale image
    if im_content.shape[2] != 1:
        # Convert to grayscale
        print('The provided field is RGB. Converting to grayscale for propagation.')
        im_content = convert_to_grayscale(im_content, method = 'weighted')
    
    # Remove 3rd dimension
    im_content = np.squeeze(im_content)

    # If the data is complex, get the absolute value (we do thresholding on the intensity)
    im_content = np.abs(im_content)

    if plot_process:
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.imshow(im_content,  origin='lower', interpolation='antialiased')
        ax.set_title('Provided image')

    # ----------- THRESHOLDING ------------
    if method == 'minimum':
        th = threshold_minimum(im_content)
    elif method == 'isodata':
        th = threshold_isodata(im_content)
    elif method == 'li':
        th = threshold_li(im_content)
    elif method == 'mean':
        th = threshold_mean(im_content)
    elif method == 'otsu':
        th = threshold_otsu(im_content)
    elif method == 'triangle':
        th = threshold_triangle(im_content)
    elif method == 'yen':
        th = threshold_yen(im_content)
    elif method == 'local':
        th = threshold_local(im_content) 
    else:
        raise ValueError('The specified thresholding method %s does not exist.' % method)

    # Get the support_mask. We want this mask to be zero where there is an 'object', and 1 otherwise.
    support_mask = im_content > th if obj_color == 'dark' else im_content < th

    if plot_process:
        ax = fig.add_subplot(132)
        ax.imshow(support_mask,  origin='lower', interpolation='antialiased')
        ax.set_title('Mask')

    # Make the masks larger if indicated
    if pad > 0:
        # The dilation algorithm makes the bright regions larger, so we need to reverse the mask for the dilation operation
        support_mask = 1 - support_mask
        # Sel array indicates what is considered 'neighbors'
        sel_array = np.ones((pad+1, pad+1))
        support_mask = binary_dilation(support_mask, sel_array)

        # go back to the right shape
        support_mask = 1 - support_mask
    
    if plot_process:
        ax = fig.add_subplot(133)
        ax.imshow(support_mask,  origin='lower', interpolation='antialiased')
        ax.set_title('Dilated Mask')
    
    return support_mask

    

