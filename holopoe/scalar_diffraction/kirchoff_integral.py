# Propagates holograms by directly calculating the Fresnel-Kirchoff integral.
# This is the most accurate method, but also the most time consuming (i.e, unfeasible, at least in my laptop!)

import numpy as np
from holopoe.process.image_processing import convert_to_grayscale
from holopoe.image import HPOEImage
from holopoe.errors import MissingParameter

def propagate_kirchoff(field, d, output_pixel_spacing, output_num_pixels, ref_wave='plane', approx_angle=3, L=None):

    """
    Propagates a field along the optical axis evaluating directly the Kirchoff integral.

    Remember: holopoe is agnostic to units, and the propagation result will be
    correct as long as the distance and wavelength and pixel spacing are in the same units.

    Parameters
    ----------
    field : HPOEImage
       Field to propagate
    d : float or list of floats
       Distance(s) to propagate.  A list tells to
       propagate to several distances and returns a list of HPOEImages at each z
    output_pixel_spacing: two element list of floats
        desired pixel spacing for the output image
    output_num_pixels: two element list of floats
        desired number of pixels for the output image
    ref_wave: string, either 'plane' or 'spherical'
    approx_angle: int, either 0, 1 or 2
        In the Fresnel-Kirchoff integral there is a term that has to do with the angle between the source and object and 
        the angle between the camera and the reconstruction plane. (page 33 of holography handbook)
        If 3, we approximate these two angles as being close to 0.
        If 2, we approximate the first angle (angle source-camera) as being close to 0. Equivalent to the situation where 
            the source is far from the object.
        If 1, we approximate the second angle (angle camera-reconstruction) as being close to 0. Equivalent to the situation where 
            the object and the camera are far away.
        If 0, we do not approximate any angle.
    L: float, distance source - camera.

    Returns
    -------
    A list of HPOEImages with the propagated field at the different requested distances.

    """

    # Make sure approx angle is right
    if approx_angle not in [0, 1, 2, 3]:
        raise ValueError('Approx angle has to be either 0, 1, 2 or 3.')

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

    # Container for the backpropagated images
    propagated_im = list()

    # Input image grid
    ks = np.arange(0, field.num_pixels[0], 1) - field.num_pixels[0]/2
    ls = np.arange(0, field.num_pixels[1], 1) - field.num_pixels[1]/2

    ls, ks = np.meshgrid(ls, ks)
    dx_in = field.pixel_spacing[0]
    dy_in = field.pixel_spacing[1]

    # Output image grid
    xis = np.arange(0, output_num_pixels[0], 1) - output_num_pixels[0]/2
    nus = np.arange(0, output_num_pixels[1], 1) - output_num_pixels[1]/2
    dx_out = output_pixel_spacing[0]
    dy_out = output_pixel_spacing[1]

    # Iterate over propagation distances
    for dist in d:

        # Container for the output image array
        im_array = np.zeros(output_num_pixels, dtype=complex)

        # Iterate over each output pixel
        for i, xi in enumerate(xis):

            print('%d out of %d rows done' % (i, len(xis)))
            for j, nu in enumerate(nus):
                
                rho = np.sqrt( (ks*dx_in - i*dx_out)**2 + (ls*dy_in - j*dy_out)**2 + dist**2 )

                exp_factor = np.exp( -1j * 2 * np.pi * rho / med_wavelen ) / rho

                # Calculate the angle factor depending on the approx indicated
                if approx_angle == 3:
                    # Both angles are ~ 0 --> angle_factor = cos(theta1) + cos(theta2) = 2. Since
                    # it is a constant factor we can consider it 1.
                    cos_ang1 = 1
                    cos_ang2 = 1
                elif approx_angle == 2:
                    # Angle source-camera close to 0
                    cos_ang1 = 1
                    # Angle 2: camera - reconstruction plane
                    cos_ang2 = dist / np.sqrt( dist**2 + (xi*dx_out - ks*dx_in)**2 + (nu*dy_out - ls*dy_in)**2 )
                elif approx_angle == 1:
                    # Angle camera-reconstruction plane close to 0
                    cos_ang2 = 1
                     # Angle 1: source - camera
                    if ref_wave == 'plane':
                        # If it's a plane wave, this angle is 0, so its cosine is 1
                        cos_ang1 = 1
                    else:
                        # Calculate the cos(ang1) as the dot product of the vector normal to the screen and 
                        # the vector source-screen. If you do the math, this results in cos(angle) = L/rho
                        cos_ang1 = L/rho
                elif approx_angle == 0:
                    # No angular approximation
                    # Angle 1: source - camera
                    if ref_wave == 'plane':
                        # If it's a plane wave, this angle is 0, so its cosine is 1
                        cos_ang1 = 1
                    else:
                        # Calculate the cos(ang1) as the dot product of the vector normal to the screen and 
                        # the vector source-screen. If you do the math, this results in cos(angle) = L/rho
                        cos_ang1 = L/rho
                    # Angle 2: camera - reconstruction plane
                    cos_ang2 = dist / np.sqrt( dist**2 + (xi*dx_out - ks*dx_in)**2 + (nu*dy_out - ls*dy_in)**2 )
            
                angle_factor = cos_ang1 + cos_ang2

                im_array[i,j] = np.sum(field * angle_factor * exp_factor)


        propagated_im.append(HPOEImage(path=None, image=im_array, pixel_spacing=output_pixel_spacing,
             n_medium=field.n_medium, wav=field.wav))


