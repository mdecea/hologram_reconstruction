import numpy as np

def norm_ft_coords(pixel_spacing, num_pixels, shift=True):
    """
    Returns the normalixed frequency coordinates of the fourier transform, 
    which are given by n/(num_pixel*pixel_spacing) for n = 0, 1, ... num_pixels - 1

    If shift is True, it returns the normalized frequency coordinates centered at 0. 
    """
    if shift:
        return np.arange(-num_pixels/2, num_pixels/2, 1)/(num_pixels*pixel_spacing)
    else:
        return np.arange(0, num_pixels, 1)/(num_pixels*pixel_spacing)


def check_conv_approach_validity(dist, med_wavelen, pixel_spacing, num_pixels, fresnel_approx):
    """
    Checks that the propagation distance is small enough so that no alisaing in the frequency domain occurs.

    See "Introduction to Modern Digitial Holography With Matlab, page 99"
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
    fresnel_approx: boolean
        if True, it also checks for the validity of the Fresnel approximation
    Returns
    -------
    A boolean indicating if the convolution approach is valid or not.

    """

    # The requirement in each direction is: z < sqrt(4*pixel_spacing^2-wav^2)*num_pixels*pixel_spacing/(2*wav)

    req_x = (dist < np.sqrt(4*pixel_spacing[0]**2-med_wavelen**2)*num_pixels[0]*pixel_spacing[0]/(2*med_wavelen))
    req_y = (dist < np.sqrt(4*pixel_spacing[0]**2-med_wavelen**2)*num_pixels[0]*pixel_spacing[0]/(2*med_wavelen))

    if (not req_x) or (not req_y):
        print('WARNING!! The required distance %.2f does not fulfill the Nyquist criterium for the convolution approach. Consisder padding the image.' % dist) 

    if (fresnel_approx):
        
        # Check validity of the Fresnel approx (z >> x,y span of the camera)
        # We assume >> is 10x, that' why there is a factor of 10 multiplying.
        req_fresnel = (dist > 10 * (1/8 * ( (num_pixels[0]*pixel_spacing[0])**2 + (num_pixels[1]*pixel_spacing[1])**2 )**2 / med_wavelen )**(1/3))

        if (not req_fresnel):
            print('WARNING!! The required distance %.2f is not long enough for the Fresnel approximation to be valid!' % dist) 