# Loading and saving operations

import sys
sys.path.insert(0, '..')

from PIL import Image as pilimage
import numpy as np
import os
import glob
import matplotlib
from errors import LoadError, Error

tif_list = ['.tif', '.TIF', '.tiff', '.TIFF']

def load_image(path, channels='all'):
    """
    Load image from a path

    Parameters
    ----------
    path: string 
        the path to the dimage
    channels: int or tuple of ints or 'all' (optional)
        number(s) of channel(s) to load for a color image (in general 0=red,
        1=green, 2=blue)

    Returns
    -------
    image : 3d numpy array with the image. Dim 1 is x, Dim 2 is y, Dim 3 is color (or channel).
    """
    
    with open(path,'rb') as im:
        pi = pilimage.open(im)
        arr = np.asarray(pi).astype('d')

    # Crceate the third dimensions if there is none
    if len(arr.shape) < 3:
        arr = arr.reshape((arr.shape[0], arr.shape[1], 1))

    # Make sure channels is a list
    if channels == 'all':
        # Convert from 'all' to a list containing all the channels
        channels = list(range(arr.shape[2]))

    if not isinstance(channels, list):
        try:
            # In case channels is a tuple
            channels = list(channels)
        except:
            # In case channels is a single integer
            channels = [channels]

    # Keep the channels of interest
    if np.max(channels) >= arr.shape[2]:
        raise LoadError(path,
            "The image doesn't have a channel number {0}".format(np.max(channels)))
    else:
        arr = arr[:, :, channels]

    return arr

def load_average(filepath, channels='all', image_glob='*.tif'):
    """
    Loads and averages a set of images (usually as a background)

    Parameters
    ----------
    filepath : string or list(string)
        Directory or list of filenames or filepaths. If filename is a
        directory, it will average all images matching image_glob.
    image_glob : string
        Glob used to select images (if images is a directory)

    Returns
    -------
    average_image : 3d numpy array
        Image which is an average of images
    """

    if isinstance(filepath, str):
        if os.path.isdir(filepath):
            filepath = glob.glob(os.path.join(filepath, image_glob))
        else:
            #only a single image
            filepath=[filepath]

    if len(filepath) < 1:
        raise LoadError(filepath, "No images found")

    average_image = None

    for i, fname in enumerate(filepath):
        if i ==0:
            average_image = load_image(fname, channels)
        else:
            average_image = (average_image + load_image(fname, channels))/2
    
    return average_image

def save_image(filename, image):
    """Save an Imahe object or numpy array as an image. It defaults to tiff.

    Parameters
    ----------
    filename : basestring
        filename in which to save image. 
    image : 3d numpy array or Image object
        image to save.
    """

    from image import HPOEImage

    # if we don't have an extension, default to tif
    if os.path.splitext(filename)[1] == '': filename += '.tif'

    # If the file extension is tif and the object is an Image, we can add metadata
    metadat = False
    if (os.path.splitext(filename)[1] in tif_list) and isinstance(image, HPOEImage):
        metadat = str(image)
        # import ifd2 - hidden here since it doesn't play nice in some cases.
        from PIL.TiffImagePlugin import ImageFileDirectory_v2 as ifd2
        tiffinfo = ifd2()
        # place metadata in the 'imagedescription' field of the tiff metadata
        tiffinfo[270] = metadat

    if isinstance(image, HPOEImage):
        # Get the numpy array only
        image = image.im
        # Holograms are grayscale images, so we only need [x, y]
        image = np.abs(image[:,:,0])

    image = image/np.max(image)

    matplotlib.pyplot.imsave(filename, image)