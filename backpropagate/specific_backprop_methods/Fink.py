# Generates a hologram with a spherical reference wave, using the approach
# described in Latychevskaia and Fink, "Practical algorithms for simulation and reconstructionof digital in-line holograms",
# Applied Optics 54 (9), 2015

# TODO: ACCOUNT FOR NON-CENTERED SPHERICAL WAVE REFERENCE!

import copy
import numpy as np
from scipy import interpolate
from image import HPOEImage
from intergrid.intergrid import Intergrid

def backprop_holo_sph_wave_Fink(holo, d, L_object, pinhole_center=None):

    """
    Simulates a hologram for a field at the aperture given by aperture_field, recorded a distance d
    from the object. Uses the approach in Latychevskaia and Fink, Applied Optics 2015.
    It assumes a spherical wave whose origin is a distance L_object away from the object.

    Remember: holopoe is agnostic to units, and the propagation result will be
    correct as long as the distance and wavelength and pixel spacing are in the same units.

    Parameters
    ----------
    holo : HPOEImage
       The recorded hologram
    d : float or list of floats
       Distance(s) at which the hologram is recorded.  A list tells to
       propagate to several distances and returns a list of HPOEImages at each z.
    L_object: float (only required for spherical wave)
        Distance between the pinhole and the object!
    pinhole_center: list of ints (only required for spherical wave)
        Position of the center of the spherical wave, in pixels. If None, we will assume that it corresponds to the
        center of the image

    Returns
    -------
    A list of HPOEImages with the generated hologram at the different requested distances

    """

    # ----- PREPROCESSING ------
    # Do this so that we do not affect the original hologram
    h_field = copy.deepcopy(holo)

    # Make sure ft has only 1 element in z, and if not convert to grayscale image
    if h_field.im.shape[2] != 1:
        # Convert to grayscale
        print('The provided hologram is RGB. Converting to grayscale for propagation.')
        h_field = convert_to_grayscale(h_field, method = 'weighted')

    # Make sure that we have all the necessary metadata
    if h_field.n_medium is None or h_field.wav is None:
        raise MissingParameter("refractive index and wavelength")

    # Convert distances to an array
    if not (isinstance(d, (list, np.ndarray))):
        d = [d]
    d = np.array(d)

    med_wav = h_field.wav / h_field.n_medium
    num_pixels = h_field.num_pixels
    pixel_spacing = h_field.pixel_spacing

    if pinhole_center is None:
        pinhole_center = [num_pixels[0]/2, num_pixels[1]/2]

    h_f = np.squeeze(h_field.im)

    # ------ ACTUAL COMPUTATION ------

    backprop_fields = list()

    for dist in d:

        # Step (a): recast hologram into kx, ky coordinates
        kx, ky = get_aperture_coords(L_object + dist, num_pixels, pixel_spacing)
        # Get the x and y values corresponding to this kx, ky coordinates
        x_mat, y_mat = convert_k_to_x(L_object + dist, kx, ky)
        ap_coords_field = resample_regular_to_irregular(h_f, pixel_spacing, x_mat, y_mat)

        # Step (b): Calculate jacobian
        ky_mat, kx_mat = np.meshgrid(ky, kx)
        jac = (L_object + dist)**2/(1-kx_mat**2-ky_mat**2)**2

        # Step (c): ift of the product
        ift_field = np.fft.ifft2(ap_coords_field*jac)
        ift_field = np.fft.ifftshift(ift_field)

        # Steps (d): Calculate phase factor
        source_xs, source_ys = get_source_plane_coords(L_object + dist, med_wav, num_pixels, pixel_spacing)
        ph_factor = np.exp(- 1j * (np.pi / (med_wav * dist)) * ( source_xs**2 + source_ys**2 ) ) 

        # Steps (e, f): Calculate fft of (c) and (d)
        #backprop_field = np.fft.fft2(ift_field*ph_factor) 
        backprop_field = np.fft.fft2(np.fft.fftshift(ift_field*ph_factor) ) 

        # Add the 3rd dimension to have an HPOEImage
        backprop_field = backprop_field.reshape((backprop_field.shape[0], backprop_field.shape[1], 1))

        # The pixel spacing is different!
        backprop_pix_spacing = get_backprop_pixel_spacing(L_object + dist, L_object, num_pixels, pixel_spacing)
    
        backprop_fields.append(HPOEImage(path=None, image=backprop_field, pixel_spacing=backprop_pix_spacing,
             n_medium=h_field.n_medium, wav=h_field.wav))

    return backprop_fields

def get_backprop_pixel_spacing(total_prop_length, L_object, num_pixels, pixel_spacing):

    Lx, Ly = num_pixels[0]*pixel_spacing[0], num_pixels[1]*pixel_spacing[1]

    kx_max = (Lx/2) / np.sqrt((Lx/2)**2 + total_prop_length**2)
    ky_max = (Ly/2) / np.sqrt((Ly/2)**2 + total_prop_length**2)
    delta_kx = 2*kx_max/num_pixels[0]
    delta_ky = 2*ky_max/num_pixels[1]

    return [L_object/delta_kx, L_object/delta_ky]

def get_aperture_coords(total_prop_length, num_pixels, pixel_spacing):
    """
    Returns the 'aperture coordinates' in which the field at the camera plane is first calculated. 

    Parameters
    ----------
    total_prop_length : float
       dist source-sample + dist sample-camera
    num_pixels: two element list of integers
        Number of pixels in each direction
    pixel_spacing: two element list of floats
        Pixel spacing in each direction
    wav: float
        Wavelength
    """
    
    Lx, Ly = num_pixels[0]*pixel_spacing[0], num_pixels[1]*pixel_spacing[1]

    kx_max = (Lx/2) / np.sqrt((Lx/2)**2 + total_prop_length**2)
    delta_kx = 2*kx_max/num_pixels[0]
    kx = (np.arange(0, num_pixels[0], 1) - num_pixels[0]/2)*delta_kx

    ky_max = (Ly/2) / np.sqrt((Ly/2)**2 + total_prop_length**2)
    delta_ky = 2*ky_max/num_pixels[1]
    ky = (np.arange(0, num_pixels[1], 1) - num_pixels[1]/2)*delta_ky

    return kx, ky

def convert_k_to_x(total_prop_length, kx, ky):
    """
    Converts from the vectors kx, ky to a matrix indicating the sampling points of the propagated field in normal (x,y) coordinates

    Parameters
    ----------
    total_prop_length : float
       dist source-sample + dist sample-camera
    kx, ky: list of floats
        vectors with the kx and ky coordinates
    """

    [ky_mat, kx_mat] = np.meshgrid(ky, kx)

    x_mat = total_prop_length*kx_mat / (np.sqrt(1 - kx_mat**2 - ky_mat**2) )
    y_mat = total_prop_length*ky_mat / (np.sqrt(1 - kx_mat**2 - ky_mat**2) )

    return x_mat, y_mat

def resample_regular_to_irregular(field, pixel_spacing, x_coords_mat, y_coords_mat):
    """
    Resamples the field that is know in a regularly spaced grid (set by pixel spacing) to the cooridnates given by x_coords_mat, y_coords_mat

    Parameters
    ----------
    field : 2d numpy array
       the field that we want to resample
    pixel_spacing: 2 element list of floats
        the piexl spacing at wjhich the field is taken
    x_coords_mat, y_coords_mat: 2d numpy arrays
        matrices with the x and y coordinates we want the field to be resampled at.
    """

    nx, ny = copy.deepcopy(field.shape)
    # x and y vectors of the regular grid
    x = (np.arange(0, nx, 1) - nx/2)*pixel_spacing[0]
    y = (np.arange(0, ny, 1) - ny/2)*pixel_spacing[1]

    # Create the interpolant
    interfunc = Intergrid( field, lo=[x[0], y[0]], hi=[x[-1], y[-1]] )

    query_points = [[x_val, y_val] for x_val, y_val in zip(x_coords_mat.flatten(), y_coords_mat.flatten())]
    resampled_field = interfunc.at( query_points )
    resampled_field = resampled_field.reshape((nx, ny))

    return resampled_field

def get_source_plane_coords(total_prop_length, med_wav, num_pixels, pixel_spacing):
    """
    Calculates the coordinates at the source plane

    Parameters
    ----------
   L_object : float
       distance source-sample
    med_wav: float
        wavelength
    num_pixels: 2 element list of floats
        number of pixels in each direction
    pixel_spacing: 2 element list of floats
        the pixel spacing
    """

    # We first need delta_kx and delta_ky
    Lx, Ly = num_pixels[0]*pixel_spacing[0], num_pixels[1]*pixel_spacing[1]
    kx_max = (Lx/2) / np.sqrt((Lx/2)**2 + total_prop_length**2)
    delta_kx = 2*kx_max/num_pixels[0]
    ky_max = (Ly/2) / np.sqrt((Ly/2)**2 + total_prop_length**2)
    delta_ky = 2*ky_max/num_pixels[1]

    deltax = med_wav/(delta_kx*num_pixels[0])
    deltay = med_wav/(delta_ky*num_pixels[1])

    x_vec = (np.arange(0, num_pixels[0], 1) - num_pixels[0]/2) * deltax
    y_vec = (np.arange(0, num_pixels[1], 1) - num_pixels[1]/2) * deltay

    y_mat, x_mat = np.meshgrid(y_vec, x_vec)

    return x_mat, y_mat