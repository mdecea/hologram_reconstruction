import sys
sys.path.insert(0, "../")

from inout.visualize import plot_image

import numpy as np
import matplotlib.pyplot as plt

def sph_wave(L, pinhole_center, pixel_spacing, num_pixels, med_wavelen):
    """
    Calculates the spherical wave field at the imaging plane

    Parameters
    ----------
    L: float
        propagation distance from the wave origin

    pinhole_center: list of ints
        location of the center of the pinhole in pixels. If None, we will assume it is at the
        center of the image
    
    pixel_spacing: two element list of floats
        pixel spacing in x, y for the camera
    
    num_pixels: two element list of int
        number of pixels in x and y

    med_wavelen: float
        wavelength of the medium

    """

    x = np.arange(0, num_pixels[0], 1)
    y = np.arange(0, num_pixels[1], 1)

    y, x = np.meshgrid(y, x)

    if pinhole_center is None:
        pinhole_center = [num_pixels[0]/2, num_pixels[1]/2]

    r = np.sqrt(L**2 + (x-pinhole_center[0])**2 * pixel_spacing[0]**2 + (y-pinhole_center[1])**2 * pixel_spacing[1]**2)

    sph_mat = (1/r) * np.exp( -1j * 2 * np.pi * r / med_wavelen)  

    # Add the 3rd dimension to be compatible with HPOEImages
    sph_mat = sph_mat.reshape((sph_mat.shape[0], sph_mat.shape[1], 1))

    return sph_mat

if __name__ == '__main__':

    # Let's make some tests

    s_wav1 = sph_wave(L=5e3, pinhole_center=None, pixel_spacing=[3.8, 3.8], num_pixels=[500, 500], med_wavelen=1.13)
    #s_wav2 = sph_wave(5e3, [25, 75], [0.1, 0.1], [1000, 1000], 0.47)

    # Plot intensity and phase
    plot_image(s_wav1, axis_units = 'pixel', scaling='auto', title = None, mode='intensity_and_phase')
    #plot_image(s_wav2, axis_units = 'pixel', scaling='auto', title = None, mode='intensity_and_phase')
    #plt.show(block=True)

    plt.figure()

    plt.contour(np.angle(s_wav1[:,:,0]), 5)
    plt.show(block=True)