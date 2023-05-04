
# Propagation of fields using a non-paraxial convolution approach
# Reference: M. D. Feit and J. A. Fleck, Jr., "Beam nonparaxiality, filament formation, and beam breakup
# in the self-focusing of optical beams", J. Opt. Soc. Am. B 1988. 
# Link: http://www.mfeit.net/physics/papers/josab588.pdf

from holopoe.process.image_processing import convert_to_grayscale
from holopoe.image import HPOEImage
from holopoe.errors import MissingParameter
import numpy as np
import holopoe.scalar_diffraction.utils as utils

def propagate_conv_non_paraxial(field, d, cfsp=0, gradient_filter=False):

    """
    Propagates the given field along the optical axis using a non-paraxial convolution approach.

    Remember: holopoe is agnostic to units, and the propagation result will be
    correct as long as the distance and wavelength and pixel spacing are in the same units.

    Parameters
    ----------
    field : HPOEImage
        Field to propagate
    d : float or list of floats
       Distance(s) to propagate.  A list tells to
       propagate to several distances and returns a list of HPOEImages at each z
    cfsp : integer (optional)
       Cascaded free-space propagation factor.  If this is an integer
       > 0, the transfer function G will be calculated at d/cfsp and
       the value returned will be G**csf.  This helps avoid artifacts
       related to the limited window of the transfer function (page 166 holography handbook)
    gradient_filter : float (optional)
       For each distance, compute a second propagation a distance
       gradient_filter away and subtract.  This enhances contrast of
       rapidly varying features.  You may wish to use the number that is
       a multiple of the medium wavelength (wav / n_medium)

    Returns
    -------
    A list of HPOEImages with the propagated image at the different requested distances

    """

    # ------- PREPROCESSING ---------------
    # Make sure ft has only 1 element in z, and if not convert to grayscale image
    if field.im.shape[2] != 1:
        # Convert to grayscale
        print('The provided field is RGB. Converting to grayscale for propagation.')
        field = convert_to_grayscale(field, method = 'weighted')

    # Make sure that we have all the necessary metadata
    if field.n_medium is None or field.wav is None:
        raise MissingParameter("refractive index and wavelength")

    # Wavelength in the propagation medium
    med_wavelen = field.wav / field.n_medium

    # Convert distances to an array
    if not (isinstance(d, (list, np.ndarray))):
        d = [d]
    d = np.array(d)

    # Remove the z dimension (of length one) for the convolution method
    conv_im = np.squeeze(field.im)

    # In the convolution approach, we only need to calculate the transfer function of the system (G), 
    # and then the image is simply image = ifft(fft(field)*G).
    propagated_im = list()

    # ------- CALCULATION ------------------

    for i, dist in enumerate(d):
        
        # Check that the propagation distance is small enough to satisfy spectral sampling requirements
        # (see "Introduction to Modern Digitial Holography With Matlab, page 99")
        if cfsp > 0:
            utils.check_conv_approach_validity(dist/cfsp, med_wavelen, field.pixel_spacing, field.num_pixels, fresnel_approx=False)
        else:
            utils.check_conv_approach_validity(dist, med_wavelen, field.pixel_spacing, field.num_pixels, fresnel_approx=False)

        G = trans_func_non_paraxial(dist, field.pixel_spacing, field.num_pixels, med_wavelen, cfsp=cfsp, 
                                    gradient_filter=gradient_filter)

        ft = np.fft.fft2(conv_im)
        ft = np.fft.fftshift(ft)
        ft_g = np.fft.ifftshift(ft*G)
        res = np.fft.ifft2(ft_g)

        # Add the 3rd dimension to have an HPOEImage
        res = res.reshape((res.shape[0], res.shape[1], 1))
        propagated_im.append(HPOEImage(path=None, image=res, pixel_spacing=field.pixel_spacing,
             n_medium=field.n_medium, wav=field.wav))

    return propagated_im


def trans_func_non_paraxial(d, pixel_spacing, num_pixels, med_wavelen, cfsp=0, gradient_filter=0):
    """
    Calculates the optical transfer function to use in reconstruction with he convolution approach.
    It uses the analytical form of the transfer function in the frequency domain.

    Parameters
    ----------
    d : float or list of floats
       Reconstruction distance.  If list or array, this function will
       return an array of transfer functions, one for each distance
    pixel_spacing : two element list of floats
       pixel spacing in the x and y directions of the camera used to acquire the field
    num_pixels: two element list of ints
        number of pixels in the x and y direction
    med_wavelen : float
       The wavelength in the medium the light is propagating through
    cfsp : integer (optional)
       Cascaded free-space propagation factor.  If this is an integer
       > 0, the transfer function G will be calculated at d/csfp and
       the value returned will be G**csf
    gradient_filter : float (optional)
       Subtract a second transfer function a distance gradient_filter
       from each z

    Returns
    -------
    trans_func : list of 2D numpy arrays
       The calculated transfer function at each distance d.

    """

    if (cfsp > 0):
        cfsp = int(abs(cfsp))  # should be nonnegative integer
        d = d / cfsp

    # Calculate the normalized frequency indices
    m = utils.norm_ft_coords(pixel_spacing[0], num_pixels[0]) #- num_pixels[0]/2  # m = (0, 1, ... N-1)/(N*deltaX)
    n = utils.norm_ft_coords(pixel_spacing[1], num_pixels[1]) #- num_pixels[1]/2 # n = (0, 1, ... M-1)/(M*deltaX)

    n, m = np.meshgrid(n, m)

    # -----------

    # Non paraxial case
    k = 2*np.pi/med_wavelen
    
    g = np.exp( -1j * d * ( ( (2*np.pi*m)**2 + (2*np.pi*n)**2 ) / (np.sqrt(k**2 - (2*np.pi*m)**2 - (2*np.pi*n)**2) + k )))

    if gradient_filter:
        g -= np.exp( -1j * (d + gradient_filter) * ( ( (2*np.pi*m)**2 + (2*np.pi*n)**2 ) / (np.sqrt(k**2 - (2*np.pi*m)**2 - (2*np.pi*n)**2) + k )))

    if cfsp > 0:
        g = g ** cfsp

    return g