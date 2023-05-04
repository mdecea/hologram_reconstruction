# Collection of image processing operations that can be done to a HPOEImage object

import sys

from holopoe.errors import BadImage
import numpy as np
from scipy.signal import detrend as detrend_scipy
from scipy.ndimage import gaussian_filter
from holopoe.image import HPOEImage
from holopoe.inout.visualize import ImageForCoordExtraction
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle as sk_denoise
import copy

def normalize(img):
    """
    Normalize an image by dividing by the pixel average.
    This gives the image a mean value of 1.

    Parameters
    ----------
    img : HPOEImage or 3D numpy array
       The array to normalize

    Returns
    -------
    A new HPOEImage object with the normalized data
    """

    image = copy.deepcopy(img)

    if isinstance(image, HPOEImage):
        im_content = image.im
    else:
        im_content = image
    
    for i in range(im_content.shape[2]):
        channel_av = np.average(im_content[:,:,i])
        im_content[:,:,i] = im_content[:,:,i]/channel_av
    
    if isinstance(image, HPOEImage):
        image.im = im_content
    else:
        image = im_content

    return image

def get_line(image, direction, extent=None, interactive=True, coord=None):
    """"
    Plot line profile along a specified direction.

    If interactive = True, it makes the user select a pixel, and we will create a new plot showing the pixels along
        the direction indicated (either 'x' or 'y').
    If interactive = False, it uses the coordinate indicated in coord. coord can either be a list (indicating x,y)
    coordinates of the center, or a single number (which will be used as the y position if direction is 'x' or
    the x position if direction is 'y') 

    If extent is indicated, instead if plotting along the whole line, it plots from the [center-extent] to [center+extent]
    """

    if interactive:
        c_ext = ImageForCoordExtraction()
        coords = c_ext.get_coord(image)
    else:
        if isinstance(coord, list):
            coords = coord
        else:
            coords = (coord, coord)

    if direction == 'x':
        c = int(coords[0])
        if extent is None:
            line = image.im[:, c]
        else:
            c2 = int(coords[1])
            line = image.im[(c2-extent):(c2+extent), c]

    elif direction == 'y':
        c = int(coords[1])
        if extent is None:
            line = image.im[c, :]
        else:
            c2 = int(coords[0])
            line = image.im[c, (c2-extent):(c2+extent)]

    else:
        print('Direction not recognized, should be either x or y. Doing nothing.')
        return

    if interactive:
        plt.figure()
        plt.plot(line)
        plt.show(block=False)
    return line

def remove_average(image):
    """
    Removes the pixel average from an image. It does each color channel separately

    Parameters
    ----------
    image : HPOEImage or 3D numpy array
       The array to normalize

    Returns
    -------
    A new HPOEImage object with the normalized data
    """

    img = copy.deepcopy(image)
    if isinstance(image, HPOEImage):
        im_content = img.im
    else:
        im_content = img

    for i in range(im_content.shape[2]):
        im_content[:,:,i] = im_content[:,:,i] - np.average(im_content[:,:,i])
    
    if isinstance(image, HPOEImage):
        img.im = im_content
        return img
    else:
        return im_content

def detrend(image):
    '''
    Remove linear trends from an image. It does each color channel separately

    Performs a 2 axis linear detrend using scipy.signal.detrend

    Parameters
    ----------
    image : HPOEImage or 3D nummpy array. 
       Image to process

    Returns
    -------
    A new HPOEImage or 3D numpy arrat object with the detrended data in x and y
    '''

    img = copy.deepcopy(image)
    if isinstance(image, HPOEImage):
        im_content = img.im
    else:
        im_content = img

    det_im = detrend_scipy(detrend_scipy(im_content, 0), 1)

    if isinstance(image, HPOEImage):
        img.im = det_im
        return img
    else:
        return det_im

def zero_filter(image):
    '''
    Search for and interpolate pixels equal to 0.
    This is to avoid NaN's when a hologram is divided by a BG with 0's.

    Parameters
    ----------
    image : HPOEImage or 3D numpy array
       Image to process

    Returns
    -------
    Another HPOEImage or 3D numpy array where pixels = 0 are instead given values equal to average of
       neighbors.
    '''

    img = copy.deepcopy(image)
    if isinstance(image, HPOEImage):
        im_content = img.im
    else:
        im_content = img

    output = im_content.copy()

    # Iterate 
    zero_pix = np.where(im_content == 0)  # Returns [[x_0pixel1, x_0pixel2, ...],[y_0pixel1, y_0pixel2, ...], [channel_0pixel1, ...]]

    # check to see if adjacent pixels are 0, if more than 1 dead pixel
    #if len(zero_pix[0]) > 1:
    #    delta_rows = zero_pix[0] - np.roll(zero_pix[0], 1)
    #    delta_cols = zero_pix[1] - np.roll(zero_pix[1], 1)
    #    delta_channel = zero_pix[2] - np.roll(zero_pix[2], 1)

    #    if 0 in delta_channel[np.where(delta_rows[np.where(delta_cols ==0)] == 1)]:
            # Human readable: check if the dark pixels that are in 
            # consecutive columns (delta_cols == 0) are in adjacent rows (delta_rows == 1) and in the same channel (delta_channel == 0)
    #        raise BadImage('Image has adjacent dead pixels, cannot remove dead pixels')
        
    #    if 0 in delta_channel[np.where(delta_cols[np.where(delta_rows ==0)] == 1)]:
            # Human readable: check if the dark pixels that are in 
            # consecutive rows (delta_rows == 0) are in adjacent columns (delta_cols == 1) and in the same channel (delta_channel == 0)
    #        raise BadImage('Image has adjacent dead pixels, cannot remove dead pixels')


    for row, col, chan in zip(zero_pix[0], zero_pix[1], zero_pix[2]):
        # in the bulk
        if ((row > 0) and (row < (im_content.shape[0]-1)) and
            (col > 0) and (col < im_content.shape[1]-1)):
            output[row, col, chan] = np.sum(im_content[row-1:row+2, col-1:col+2, chan]) / 8.

        else: # deal with edges by padding through mirroring
            padded_im = np.ones((im_content.shape[0]+2, im_content.shape[1]+2))
            padded_im[1:-1, 1:-1] = im_content[:,:, chan]
            padded_im[0, 1:-1] = im_content[0, :, chan]
            padded_im[-1, 1:-1] = im_content[-1, :, chan]
            padded_im[1:-1, 0] = im_content[:, 0, chan]
            padded_im[1:-1, -1] = im_content[:, -1, chan]

            if row == 0:
                if (col > 0) and (col < im_content.shape[1]-1):
                    output[row, col] = np.sum(padded_im[row:row+3, col-1:col+2]) / 8.
                elif col == 0:
                    output[row, col] = np.sum(padded_im[row:row+3, col:col+3]) / 8.
                else:
                    output[row, col] = np.sum(padded_im[row:row+3, col-2:col+1]) / 8.

            elif row == (im_content.shape[0]-1):
                if (col > 0) and (col < im_content.shape[1]-1):
                    output[row, col] = np.sum(padded_im[row-2:row+1, col-1:col+2]) / 8.
                elif col == 0:
                    output[row, col] = np.sum(padded_im[row-2:row+1, col:col+3]) / 8.
                else:
                    output[row, col] = np.sum(padded_im[row-1:row+1, col-2:col+1]) / 8.
            elif col == 0:
                output[row, col] = np.sum(padded_im[row-1:row+2, col:col+3]) / 8.
            else:
                output[row, col] = np.sum(padded_im[row-1:row+2, col-2:col+1]) / 8.

        # print('Pixel with value 0 reset to nearest neighbor average')

    # If there are many zero pixels close together, this simple interpolation won't work. To avoid having nans, 
    # substitue the remaining 0 pixels by the average
    zero_pix = np.where(output == 0)
    im_av = np.average(output)
    if len(zero_pix[0]) > 0:
        print('Setting %d pixels to the average of the image' % len(zero_pix[0]))
        for row, col, chan in zip(zero_pix[0], zero_pix[1], zero_pix[2]):
            output[row,col,chan] = im_av

    if isinstance(img, HPOEImage):
        img.im = output
        return img
    else:
        return output

def add_noise(image, noise_mean=.1, smoothing=.01, poisson_lambda=1000):
    """Add simulated noise to images. Intended for use with exact
    calculated images to make them look more like noisy 'real'
    measurements.

    Real image noise usually has correlation, so we smooth the raw
    random variable. The noise_mean can be controlled independently of
    the poisson_lambda that controls the shape of the distribution. In
    general, you can stick with our default of a large poisson_lambda
    (ie for imaging conditions not near the shot noise limit).

    Defaults are set to give noise vaguely similar to what we tend to
    see in our holographic imaging.

    Parameters
    ----------
    image : HPOEImage or 3D numpy array
        The image to add noise to.
    noise_mean: mean power(?) of the noise with respect to the mean of the image
    smoothing : float
        Fraction of the image size to smooth by. Should in general be << 1
    poisson_lambda : float
        Used to compute the shape of the noise distribution. You can generally
        leave this at its default value unless you are simulating shot noise
        limited imaging.

    Returns
    -------
        A copy of the input image with noise added (as an HPOEImage object)

    """

    img = copy.deepcopy(image)

    if isinstance(img, HPOEImage):
        im_content = img.im
    else:
        im_content = img

    raw_poisson = np.random.poisson(poisson_lambda, im_content.shape)
    smoothed = gaussian_filter(raw_poisson, np.array(im_content.shape)*smoothing)
    noise = smoothed/smoothed.mean() * noise_mean * np.average(im_content)

    noisy_im = im_content + noise

    if isinstance(img, HPOEImage):
        p_sp, n, wav = image.get_metadata()
        return HPOEImage(path=None, image=noisy_im, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
    else:
        return noisy_im

def bg_correct(image, background, dark_image = None, operation = 'subtract', max_ratio = 3):
    """
    Corrects the hologram for background illumination and residual room illumination.

    Parameters
    ----------
    image : HPOEImage
        Image to be background divided.
    background : HPOEImage
        background image recorded with the same optical setup.
    dark_image : HPOEImage (optional)
        dark field image recorded without illumination.
    operation: either 'subtract' (image-background) , 'divide' (image - dark_image)/(background-dark_image), 'divide_and_subtract' (image/background - 1)
    max_ratio: float (optional)
        when using a background correction that requires dividing, it limits the maximum ratio between the hologram and the background image to max_ratio.
        This is to avoid instances where, due to noise, we have an unusually large ratio between the background and the hologram

    Returns
    -------
       A copy of the background-corrected input image.

    """

    bg_corrected_image = None

    if operation == 'subtract':
        bg_corrected_image = image - background

    elif operation == 'divide' or operation == 'divide_and_subtract':

        # Make sure there are no dark pixels, because if not that will fuck up the data and generate nans
        if dark_image is None:
            bg_corrected_image = image/zero_filter(background)
        else:
            bg_corrected_image = (image-dark_image)/zero_filter(background-dark_image)

        # Limit ratio
        if isinstance(bg_corrected_image, HPOEImage):
            indices = np.where(bg_corrected_image.im > max_ratio)
            bg_corrected_image.im[indices] = max_ratio
        else:
            indices = np.where(bg_corrected_image > max_ratio)
            bg_corrected_image[indices] = max_ratio

        if operation == 'divide_and_subtract':

            bg_corrected_image = bg_corrected_image - 1

    else:
        raise ValueError('The specified operation for background removal is not recognized.')

    return bg_corrected_image
   
def subimage(image, center = None, span = None, init = None, end = None):
    """
    Pick out a region of an image.

    We can specify either as (center and span) or (init and end). If init is not None,
    we will assume the desired is (init and end).

    (center and span): we will select elements between center-span/2 and center+span/2 (in each dimension)
    (init and end): we will select elements between init and end (in each dimension)

    Parameters
    ----------
    image : HPOEIage or 3D numpy array
        The array to subimage
    center : list of ints or floats
        The desired center of the region, should have the same number of
        elements as the arr has dimensions. Floats will be rounded.
    span : int or list of ints
        Desired span of the region to be picked around the center.  If a single int is given the region will
        be that dimension along every axis.
    init : list of ints
        The desired start indices of the regoin, should have the same number of
        elements as the arr has dimensions. Floats will be rounded.
    end : list of ints
        The desired end indices of the regoin, should have the same number of
        elements as the arr has dimensions. Floats will be rounded.

    Returns
    -------
    A new HPOEImage wiht the region of interest.
    """

    img = copy.deepcopy(image)

    if isinstance(img, HPOEImage):
        im_content = img.im
    else:
        im_content = img

    rows, cols, chans = im_content.shape

    if init is not None:
        # (init, end) method
        # Support the fact that no channel data is given

        # Make sure that the bounds are within the image size
        if any(np.array(init) < 0):
            raise ValueError('Specified subimage with negative bounds.')
        if end[1] > rows or end[0] > cols:
            raise ValueError('The specified subimage dimensions are larger than the original image.')

        if len(init) < 3:
            new_im = im_content[init[1]:end[1], init[0]:end[0], :]
        else:
            if end[2] > chans:
                raise ValueError('The specified subimage dimensions are larger than the original image.')
            new_im = im_content[init[1]:end[1], init[0]:end[0], init[2]:end[2]]

    else:

        # Make sure that the bounds are within the image size
        if (center[1] + int(span[1]/2)) > rows or (center[0] + int(span[0]/2)) > cols:
            raise ValueError('The specified subimage dimensions are larger than the original image.')


        # (center, span) method
        if len(center) < 3:
            new_im = im_content[(center[1] - int(span[1]/2)):(center[1] + int(span[1]/2)), 
                                (center[0] - int(span[0]/2)):(center[0] + int(span[0]/2)),
                                :]
        else:
            new_im = im_content[(center[1] - int(span[1]/2)):(center[1] + int(span[1]/2)), 
                                (center[0] - int(span[0]/2)):(center[0] + int(span[0]/2)),
                                (center[2] - int(span[2]/2)):(center[2] + int(span[2]/2))]

    if isinstance(img, HPOEImage):
        p_sp, n, wav = image.get_metadata()
        return HPOEImage(path=None, image=new_im, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
    else:
        return new_im

def pad_image(image, pad_size, val=None):
    """
    Pads the image with zero pixels. Pad_size should be even, since we pad symmetrically.

    image : HPOEIage
        The array to subimage
    pad_size: 2 element list of floats
        The number of pixels to add in each dimension (x and y). Should be an even number.\
    val: float
        The value to pad the image with. If not indicated we assume it is 0.
    """

    img = copy.deepcopy(image)

    if isinstance(img, HPOEImage):
        im_cnt = img.im
    else:
        im_cnt = img

    if pad_size[0] % 2 != 0 or pad_size[1] % 2:
        raise ValueError("Padding sizes should be even")

    if val is None:
        padded_im = np.zeros( (im_cnt.shape[0]+pad_size[0], im_cnt.shape[1]+pad_size[1], im_cnt.shape[2]) )
    else:
        print('here')
        padded_im = np.ones( (im_cnt.shape[0]+pad_size[0], im_cnt.shape[1]+pad_size[1], im_cnt.shape[2]) ) * val
    
    padded_im[ int(pad_size[0]/2) : int(-pad_size[0]/2), int(pad_size[1]/2) : int(-pad_size[1]/2), :] = im_cnt

    if isinstance(img, HPOEImage):
        p_sp, n, wav = image.get_metadata()
        return HPOEImage(path=None, image=padded_im, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
    else:
        return padded_im

def rotate(image, angle):
    """
    Rotates the image by 90 or -90 degrees. 90 degrees is ccw rotation.
    """

    img = copy.deepcopy(image)

    if isinstance(img, HPOEImage):
        im_cnt = img.im
    else:
        im_cnt = img

    if angle not in [-90, 90]:
        raise ValueError('Specified rotation angle not supported. Can only do -90 or 90 degrees.')
    
    if angle == 90:
        im = np.rot90(im_cnt)
    else:
        im = np.rot90(im_cnt, -1)

    if isinstance(img, HPOEImage):
        p_sp, n, wav = image.get_metadata()
        return HPOEImage(path=None, image=im, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
    else:
        return im

def flip(image, direction='x'):
    """
    Flips the image in either 'x' or 'y'
    """

    img = copy.deepcopy(image)

    if isinstance(img, HPOEImage):
        im_cnt = img.im
    else:
        im_cnt = img

    if direction not in ['x', 'y']:
        raise ValueError('Specified flip direction not supported. Can only do x or y.')
    
    if direction == 'x':
        im = np.fliplr(im_cnt)
    else:
        im = np.flipud(im_cnt)

    if isinstance(img, HPOEImage):
        p_sp, n, wav = image.get_metadata()
        return HPOEImage(path=None, image=im, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
    else:
        return im

def convert_to_grayscale(image, method = 'weighted'):
    """
    Converts an HPOEimage or 3d numpy array with more than 1 color channel to a grayscale image.

    Method specifies which method to use to convert to grayscale. Supoprted possibilities:
    - weighted (default): grayscale image = (av_ch1/(av_ch1+av_ch2+...))*ch1 +  (av_ch2/(av_ch1+av_ch2+...))*ch2 + ...
    - average: grayscale image = (ch1 + ch2 + ch3 + ...) / num_ch
    - luminance: correct for luminance perception of the eyes in color (only applies to RGB). 
        image = 0.2126*red + 0.7152*green + 0.0722*blue

    """

    img = copy.deepcopy(image)

    if isinstance(img, HPOEImage):
        im_cnt = img.im
    else:
        im_cnt = img

    if im_cnt.shape[2] == 1:
        print('The image is already grayscale. Doing nothing')
        return img
    
    if method == 'weighted':

        av_ch = list()
        for i in range(im_cnt.shape[2]):
            av_ch.append(np.average(im_cnt[:,:,i]))
        
        gray_image = np.zeros(im_cnt.shape[0:2])

        for i in range(im_cnt.shape[2]):
            gray_image = (av_ch[i]/(np.sum(np.array(av_ch)))) * im_cnt[:,:,i]


    elif method == 'average':
        gray_image = np.average(im_cnt, axis=2)

    elif method == 'luminance':
        if im_cnt.shape[2] != 3:
            raise BadImage('The image for grayscale conversion is not RGB. Luminance method does not apply.')
        gray_image = 0.2126*im_cnt[:,:,0] + 0.7152*im_cnt[:,:,1] + 0.0722*im_cnt[:,:,2]

    else:
        raise ValueError('The specified grayscale conversion method is not recognized.')
    
    # Convert to 3D array
    gray_image = gray_image.reshape((gray_image.shape[0], gray_image.shape[1], 1))

    if isinstance(img, HPOEImage):
        p_sp, n, wav = image.get_metadata()
        return HPOEImage(path=None, image=gray_image, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
    else:
        return gray_image
          
def denoise(image, weight=0.1, eps=0.0002):
    """
    Denoises an image (HPOEImage or 3d numpy array) using total variation denoising (provided by scikit-image).
    DOe sit separately per channel

    weight: float, optional
        Denoising weight. The greater weight, the more denoising (at the expense of fidelity to input).
    eps: float, optional
        Relative difference of the value of the cost function that determines the stop criterion.
    """

    img = copy.deepcopy(image)

    if isinstance(img, HPOEImage):
        im_cnt = img.im
    else:
        im_cnt = image

    for i in range(im_cnt.shape[2]):
        im_cnt[:,:,i] = sk_denoise(im_cnt[:,:,i], weight, eps)

    if isinstance(img, HPOEImage):
        p_sp, n, wav = image.get_metadata()
        return HPOEImage(path=None, image=im_cnt, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
    else:
        return im_cnt
    

    
