# Backpropagates a hologram using the Fresnel approach

from holopoe.process.image_processing import convert_to_grayscale
from holopoe.backpropagate.reference_waves import sph_wave
from holopoe.scalar_diffraction.fresnel_approach import propagate_fresnel
from holopoe.errors import MissingParameter
import numpy as np
import copy
from scipy.ndimage import convolve

def backpropagate_fresnel(hologram, d, ref_wave='plane', DC_suppress=False, filt_kernel= None,
              L=None, pinhole_center=None):

    """
    Backpropagates a hologram along the optical axis using the Fresnel approach.

    Remember: holopoe is agnostic to units, and the propagation result will be
    correct as long as the distance and wavelength and pixel spacing are in the same units.

    Parameters
    ----------
    hologram : HPOEImage
       Hologram to propagate
    d : float or list of floats
       Distance(s) to propagate.  A list tells to
       propagate to several distances and returns a list of HPOEImages at each z
    ref_wave: string, either 'plane' or 'spherical'
        The type of reference wave that was used to record the hologram.
        If 'spherical', L and pinhole_center need to be provided.
    DC_suppress: boolean
        Removing the average of the recorded hologram multiplied by the reference wave can help decrease the 
        DC term (holography handbook, page 106). Set this to true if for some reason we cannot remove an experimentally
        obtained background.
    filt_kernel: numpy array
        The filter kernel to apply for DC suppression. Only used if DC_suppress is True.
        If filt_kernel = None and DC_suppress = True, the DC suppression is performed by removing the average: DC_supp = im - average(im), where im = hologram*reference.
        filt_kernel should be a matrix with the weights. 
        Popular filter kernels are:  -1/9*[[1,  1, 1],
                                           [1, -8, 1],
                                           [1,  1, 1]]

                                      [[0, -1, 0],
                                       [-1, 4, -1],
                                       [0, -1, 0]]
    L: float (only required for spherical wave)
        Distance bewteen the pinhole and the camera
    pinhole_center: list of ints (only required for spherical wave)
        Position of the center of the spherical wave, in pixels. If None, we will assume that it corresponds to the
        center of the image

    Returns
    -------
    A list of HPOEImages with the reconstructed image at the different requested distances

    """

    # We only need to multiply the hologram by the conjugate of the reference wave, and propagate it
    # using the propagate_fresnel method of the scalar_diffraction package

    # DO this so that we do not affect the original hologram
    holo = copy.deepcopy(hologram)

    # Make sure ft has only 1 element in z, and if not convert to grayscale image
    if holo.im.shape[2] != 1:
        # Convert to grayscale
        print('The provided hologram is RGB. Converting to grayscale for propagation.')
        holo = convert_to_grayscale(holo, method = 'weighted')

    # Make sure that we have all the necessary metadata
    if holo.n_medium is None or holo.wav is None:
        raise MissingParameter("refractive index and wavelength")

    if DC_suppress:
        if filt_kernel is None:
            holo.im = holo.im - np.average(holo.im)
        else:
            for dim in range(holo.im.shape[2]):
                holo.im[:,:,dim] = convolve(holo.im[:,:,dim], filt_kernel)

    if ref_wave == 'plane':
        # We do not need to do anything, since the reference wave is a constant
        pass
    elif ref_wave == 'spherical':
        # Wavelength in the propagation medium
        med_wavelen = holo.wav / holo.n_medium
        sph_wave_mat = sph_wave(L, pinhole_center, holo.pixel_spacing, holo.num_pixels, med_wavelen)
        holo.im = holo.im*np.conj(sph_wave_mat)

    return propagate_fresnel(holo, d)