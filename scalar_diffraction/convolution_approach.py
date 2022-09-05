
# Propagation of fields using the convolution approach (also named Angular Spectrum Method)

from process.image_processing import convert_to_grayscale
import scalar_diffraction.utils as utils
from image import HPOEImage
from errors import MissingParameter
import numpy as np

def propagate_conv(field, d, cfsp=0, gradient_filter=False, fresnel_approx=False):

    """
    Propagates the given field along the optical axis using the convolution approach.

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
    fresnel_approx: boolean
       If True, it uses the Fresnel approximation. Fresnel approximation holds
       when the distance object-camera (d) is much larger than the camera dimensions.
    
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
        # and fresnel if appropriate
        # (see "Introduction to Modern Digitial Holography With Matlab, page 99")
        if cfsp > 0:
            utils.check_conv_approach_validity(dist/cfsp, med_wavelen, field.pixel_spacing, field.num_pixels, fresnel_approx)
        else:
            utils.check_conv_approach_validity(dist, med_wavelen, field.pixel_spacing, field.num_pixels, fresnel_approx)

        G = trans_func(dist, field.pixel_spacing, field.num_pixels, med_wavelen, cfsp=cfsp, gradient_filter=gradient_filter, 
            fresnel_approx=fresnel_approx)

        ft = np.fft.fft2(conv_im)
        ft = np.fft.fftshift(ft)
        ft_g = np.fft.ifftshift(ft*G)
        res = np.fft.ifft2(ft_g)

        # Add the 3rd dimension to have an HPOEImage
        res = res.reshape((res.shape[0], res.shape[1], 1))
        propagated_im.append(HPOEImage(path=None, image=res, pixel_spacing=field.pixel_spacing,
             n_medium=field.n_medium, wav=field.wav))

    return propagated_im


def trans_func(d, pixel_spacing, num_pixels, med_wavelen, cfsp=0, gradient_filter=0, fresnel_approx=False):
    """
    Calculates the optical transfer function to use in reconstruction with he convolution approach.
    It uses the analytical form of the transfer function in the frequency domain.

    This routine uses the analytical form of the transfer function
    found in in Kreis [1]_.  It can optionally do cascaded free-space
    propagation for greater accuracy [2]_, although the code will run
    slightly more slowly.

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
    fresnel_approx: boolean
       If True, it uses the Fresnel approximation. Fresnel approximation holds
       when the distance object-camera (d) is much larger than the camera dimensions.

    Returns
    -------
    trans_func : list of 2D numpy arrays
       The calculated transfer function at each distance d.

    References
    ----------
    .. [1] Kreis, Handbook of Holographic Interferometry (Wiley,
       2005), equation 3.79 (page 116)

    .. [2] Kreis, Optical Engineering 41(8):1829, section 5

    """

    if (cfsp > 0):
        cfsp = int(abs(cfsp))  # should be nonnegative integer
        d = d / cfsp

    # Calculate the normalized frequency indices
    m = utils.norm_ft_coords(pixel_spacing[0], num_pixels[0]) #- num_pixels[0]/2  # m = (0, 1, ... N-1)/(N*deltaX)
    n = utils.norm_ft_coords(pixel_spacing[1], num_pixels[1]) #- num_pixels[1]/2 # n = (0, 1, ... M-1)/(M*deltaX)

    n, m = np.meshgrid(n, m)

    if not fresnel_approx:
      
      # No Fresnel approximation
      root = 1 - (med_wavelen * m) ** 2 - (med_wavelen * n) ** 2
      root *= (root >= 0)

      g = np.exp( (1j * 2 * np.pi * d / med_wavelen) * np.sqrt(root))

      if gradient_filter:
         g -= np.exp( (1j * 2 * np.pi * (d + gradient_filter) / med_wavelen) * np.sqrt(root))

      # Set the transfer function to zero where the sqrt is imaginary
      # (this is equivalent to making sure that the largest spatial
      # frequency is 1/wavelength).  (root>=0) returns a boolean matrix
      # that is equal to 1 where the condition is true and 0 where it is
      # false.  Multiplying by this boolean matrix masks the array.
      g = g * (root >= 0)
   
    else:
      # Fresnel approximation
      # From [1], page 117.

      g = np.exp(-1j * np.pi * d * ( med_wavelen * m**2 + med_wavelen * n**2  ) )

      if gradient_filter:
        g -= np.exp(1j * np.pi * (d + gradient_filter) * ( med_wavelen * m**2 + med_wavelen * n**2 ) )
    
    if cfsp > 0:
        g = g ** cfsp

    return g