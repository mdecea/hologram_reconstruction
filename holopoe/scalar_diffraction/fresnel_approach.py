# Simulates propagation using the Fresnel approximation
# This applies only when the distance sample-camera is much larger than the x-y dimensions of the camera.
# Technically, it applies when d >> (1/8 * (Lx^2 + Ly^2)^2 / wav ) ^(1/3)
# Notice how in general this is not the case for us!

from holopoe.process.image_processing import convert_to_grayscale
from holopoe.image import HPOEImage
from holopoe.errors import MissingParameter
import numpy as np

def propagate_fresnel(field, d):

    """
    Propagates a field along the optical axis using the Fresnel approximation approach.

    Remember: holopoe is agnostic to units, and the propagation result will be
    correct as long as the distance and wavelength and pixel spacing are in the same units.

    Parameters
    ----------
    field : HPOEImage
       Field to propagate
    d : float or list of floats
       Distance(s) to propagate.  A list tells to
       propagate to several distances and returns a list of HPOEImages at each z

    Returns
    -------
    A list of HPOEImages with the propagated image at the different requested distances

    """

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

    # In the Fresnel approach, we only need to calculate two phase factors P1 and P2, 
    # and then the image is simply image = ifft(field*P1)*P2
    propagated_im = list()

    for dist in d:

        # Check that the propagation distance is small enough to satisfy spectral sampling requirements
        # (see "Introduction to Modern Digitial Holography With Matlab, page 101")
        check_fresnel_approach_validity(dist, med_wavelen, field.pixel_spacing, field.num_pixels)

        P1, P2 = phase_factors(dist, field.pixel_spacing, field.num_pixels, med_wavelen) 

        ift = np.fft.ifft2(conv_im*P1)            
        res = P2*ift
        res = np.fft.fftshift(res)

        # Add the 3rd dimension to have an HPOEImage
        res = res.reshape((res.shape[0], res.shape[1], 1))

        # With the Fresnel method, the pixel spacing of the reconstructed image is different than
        # that of the acquired field
        pixel_spacing = [med_wavelen*dist/(field.num_pixels[0]*field.pixel_spacing[0]), 
                         med_wavelen*dist/(field.num_pixels[1]*field.pixel_spacing[1])]

        propagated_im.append(HPOEImage(path=None, image=res, pixel_spacing=pixel_spacing,
             n_medium=field.n_medium, wav=field.wav))

    return propagated_im

def phase_factors(d, pixel_spacing, num_pixels, med_wavelen):
    """
    Calculates the phase factors P1 and P2 to use in reconstruction with the Fresnel approach.
    P1 is the phase factor that goes inside the ifft, P2 the one that goes outside.

    Parameters
    ----------
    d : float
       Reconstruction distance.  If list or array, this function will
       return an array of transfer functions, one for each distance
    pixel_spacing : two element list of floats
       pixel spacing in the x and y directions of the camera used to acquire the field
    num_pixels: two element list of ints
        number of pixels in the x and y direction
    med_wavelen : float
       The wavelength in the medium the light is propagating through

    Returns
    -------
    trans_func : list of 2D numpy arrays
       The calculated phase factor matrices P1 and P2

    """

    # --------------- P2 calculation -----------
    # Output pixel grid
    m = np.arange(0, num_pixels[0], 1) - num_pixels[0]/2
    n = np.arange(0, num_pixels[1], 1) - num_pixels[1]/2

    n, m = np.meshgrid(n, m)

    # There is a consant factor (1j/(med_wavelen*d)) * np.exp(-1j * 2 * np.pi * d / med_wavelen) that we are not adding since it is a constant
    P2 = np.exp(- 1j * np.pi * d * med_wavelen * ( (m/(num_pixels[0]*pixel_spacing[0]))**2 + (n/(num_pixels[1]*pixel_spacing[1]))**2 ) )

    # ---------------- P1 calculation ------------
    # Input grid
    k = np.arange(0, num_pixels[0], 1) - num_pixels[0]/2
    l = np.arange(0, num_pixels[1], 1) - num_pixels[1]/2

    l, k = np.meshgrid(l, k)

    P1 = np.exp(-1j * np.pi / (med_wavelen * d) * ( (k*pixel_spacing[0])**2 + (l*pixel_spacing[1])**2) )

    return P1, P2

def check_fresnel_approach_validity(dist, med_wavelen, pixel_spacing, num_pixels):
    """
    Checks that the propagation distance is large enough so that no alisaing in the frequency domain occurs.

    See "Introduction to Modern Digitial Holography With Matlab, page 101"
    Parameters
    ----------
    dist : float
       Reconstruction distance.  
    med_wavelen: float
        Wavelength in the propagation medium
    pixel_spacing : two element list of floats
       pixel spacing in the x and y directions of the camera used to acquire the field
    num_pixels: two element list of ints
        number of pixels in the x and y direction
    Returns
    -------
    A boolean indicating if the convolution approach is valid or not.

    """

    req_x = (dist > 2*num_pixels[0]*pixel_spacing[0]**2/med_wavelen)
    req_y = (dist > 2*num_pixels[1]*pixel_spacing[1]**2/med_wavelen)

    if (not req_x) or (not req_y):
        print('WARNING!! The required distance %.2f does not fulfill the Nyquist criterium for the Fresnel approach.' % dist) 

    # Check validity of the Fresnel approx (z >> x,y span of the camera)
    # We assume >> is 10x, that' why there is a factor of 10 multiplying.
    req_fresnel = (dist > 10 * (1/8 * ( (num_pixels[0]*pixel_spacing[0])**2 + (num_pixels[1]*pixel_spacing[1])**2 )**2 / med_wavelen )**(1/3))

    if (not req_fresnel):
        print('WARNING!! The required distance %.2f is not long enough for the Fresnel approximation to be valid!' % dist) 
