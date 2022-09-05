# Generates a hologram with a spherical reference wave, using the approach
# described in Latychevskaia and Fink, "Practical algorithms for simulation and reconstructionof digital in-line holograms",
# Applied Optics 54 (9), 2015

# TODO: ACCOUNT FOR NON-CENTERED SPHERICAL WAVE REFERENCE!

import copy
import numpy as np
from scipy.interpolate import griddata
from image import HPOEImage
from process.image_processing import convert_to_grayscale
from errors import MissingParameter

def gen_holo_sph_wave_Fink(aperture_field, d, L_object, pinhole_center=None):

    """
    Simulates a hologram for a field at the aperture given by aperture_field, recorded a distance d
    from the object. Uses the approach in Latychevskaia and Fink, Applied Optics 2015.
    It assumes a spherical wave whose origin is a distance L_object away from the object.

    Remember: holopoe is agnostic to units, and the propagation result will be
    correct as long as the distance and wavelength and pixel spacing are in the same units.

    Parameters
    ----------
    aperture_field : HPOEImage
       The field at the aperture (the scattered field at the object plane)
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
    a_field = copy.deepcopy(aperture_field)

    # Make sure ft has only 1 element in z, and if not convert to grayscale image
    if a_field.im.shape[2] != 1:
        # Convert to grayscale
        print('The provided hologram is RGB. Converting to grayscale for propagation.')
        a_field = convert_to_grayscale(a_field, method = 'weighted')

    # Make sure that we have all the necessary metadata
    if a_field.n_medium is None or a_field.wav is None:
        raise MissingParameter("refractive index and wavelength")

    # Convert distances to an array
    if not (isinstance(d, (list, np.ndarray))):
        d = [d]
    d = np.array(d)

    med_wav = a_field.wav / a_field.n_medium
    num_pixels = a_field.num_pixels
    dx, dy = a_field.pixel_spacing

    if pinhole_center is None:
        pinhole_center = [num_pixels[0]/2, num_pixels[1]/2]

    # ------ ACTUAL COMPUTATION ------

    gen_holos = list()

    # Step (a) in the paper: ifft of the aperture field
    ap_ift = np.fft.ifft2(np.squeeze(a_field.im))
    ap_ift = np.fft.ifftshift(ap_ift)

    source_xs, source_ys = get_source_plane_coords(L_object, med_wav, num_pixels, [dx, dy])

    for dist in d:

        # Steps (b): Calculate phase factor
        ph_factor = np.exp(- 1j * (np.pi / (med_wav * dist)) * ( source_xs**2 + source_ys**2 ) ) 

        # Steps (c) and (d): Multiply phase factor and ifft and take the fourier transform
        field = np.fft.fft2( np.fft.fftshift(ap_ift*ph_factor) )

        # coordinates of the obtained field
        kx, ky = get_aperture_coords(L_object+dist, num_pixels, [dx, dy], L_object)
        
        # Step (e): The previous step gives the field at the camera plane in the weird coordinates (kx, ky).
        # Convert from (kx, ky) coordinates to (X, Y) coordinates (normal coordinates at the camera plane)
        x_coords_mat, y_coords_mat = convert_k_to_x(L_object + dist, kx, ky)

        # Of course, these are not points spaced regulary, so we can just resample the field in a regular grid,
        # with the same pixel size as the original image
        xy_field = resample_regular_grid(field, x_coords_mat, y_coords_mat, [dx, dy])

        # Step (f) and (g): Calculate intensity and multiply by 1/Nx*Ny
        intens = np.abs(xy_field)**2 / (num_pixels[0]*num_pixels[1])

        # Add the 3rd dimension to have an HPOEImage
        intens = intens.reshape((intens.shape[0], intens.shape[1], 1))

        # The pixel spacing is different!
        magn = (L_object + dist)/L_object
        gen_holo_spacing = [a_field.pixel_spacing[0]*magn, a_field.pixel_spacing[1]*magn]

        gen_holos.append(HPOEImage(path=None, image=intens, pixel_spacing=a_field.pixel_spacing,
             n_medium=a_field.n_medium, wav=a_field.wav))

    return gen_holos

def get_aperture_coords(total_prop_length, num_pixels, pixel_spacing, L_object):
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
    
    """
    Lx, Ly = num_pixels[0]*pixel_spacing[0], num_pixels[1]*pixel_spacing[1]

    kx_max = (Lx/2) / np.sqrt((Lx/2)**2 + total_prop_length**2)
    delta_kx = 2*kx_max/num_pixels[0]
    

    ky_max = (Ly/2) / np.sqrt((Ly/2)**2 + total_prop_length**2)
    delta_ky = 2*ky_max/num_pixels[1]
    
    """
    delta_kx = pixel_spacing[0]/L_object
    delta_ky = pixel_spacing[1]/L_object

    kx = (np.arange(0, num_pixels[0], 1) - num_pixels[0]/2)*delta_kx
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

def resample_regular_grid(field, x_coords_mat, y_coords_mat, pixel_spacing):
    """
    Resamples the field that is know at the specified coordinates to a regular grid with the specified pixel spacing

    Parameters
    ----------
    field : 2d numpy array
       the field that we want to resample
    x_coords_mat, y_coords_mat: 2d numpy arrays
        matrices with the x and y coordinates of each point.
    pixel_spacing: 2 element list of floats
        the desired pixel spacing
    """

    nx, ny = field.shape

    # Regularly spaced grid
    x_grid = (np.arange(0, nx, 1) - nx/2)*pixel_spacing[0]
    y_grid = (np.arange(0, ny, 1) - ny/2)*pixel_spacing[1]

    resampled_field = griddata((x_coords_mat.flatten(), y_coords_mat.flatten()), field.flatten(), (x_grid[None,:], y_grid[:,None]), method='cubic')

    return np.transpose(resampled_field)

def get_source_plane_coords(L_object, med_wav, num_pixels, pixel_sacing):
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

    deltax = med_wav*L_object/(num_pixels[0]*pixel_sacing[0])
    deltay = med_wav*L_object/(num_pixels[1]*pixel_sacing[1])

    x_vec = (np.arange(0, num_pixels[0], 1) - num_pixels[0]/2) * deltax
    y_vec = (np.arange(0, num_pixels[1], 1) - num_pixels[1]/2) * deltay

    y_mat, x_mat = np.meshgrid(y_vec, x_vec)

    return x_mat, y_mat