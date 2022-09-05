import sys
sys.path.insert(0, "../")

# Generates a hologram from a given intensity distribution at the aperture and reference wave.
from backpropagate.reference_waves import sph_wave
from scalar_diffraction.convolution_approach import propagate_conv
from process.image_processing import convert_to_grayscale
from scalar_diffraction.fresnel_approach import propagate_fresnel
from errors import *
import copy
import numpy as np

# Using the convolution approach
def generate_hologram_conv(aperture_field, d, ref_wave='plane', cfsp=0, 
        gradient_filter=False, fresnel_approx=False, 
        L_object=None, pinhole_center=None):

    """
    Simulates a hologram for a field at the aperture given by aperture_field, recorded a distance d
    from the object. Uses the convolution approach to calculate diffraction.

    Remember: holopoe is agnostic to units, and the propagation result will be
    correct as long as the distance and wavelength and pixel spacing are in the same units.

    Parameters
    ----------
    aperture_field : HPOEImage
       The field at the aperture (the scattered field at the object plane)
    d : float or list of floats
       Distance(s) at which the hologram is recorded.  A list tells to
       propagate to several distances and returns a list of HPOEImages at each z.
    ref_wave: string, either 'plane' or 'spherical'
        The type of reference wave that was used to record the hologram.
        If 'spherical', L and pinhole_center need to be provided.
    cfsp : integer (optional)
       Cascaded free-space propagation factor.  If this is an integer
       > 0, the transfer function G will be calculated at d/cfsp and
       the value returned will be G**csf.  This helps avoid artifacts
       related to the limited window of the transfer function
    gradient_filter : float (optional)
       For each distance, compute a second propagation a distance
       gradient_filter away and subtract.  This enhances contrast of
       rapidly varying features.  You may wish to use the number that is
       a multiple of the medium wavelength (wav / n_medium)
    fresnel_approx: boolean
       If True, it uses the Fresnel approximation. Fresnel approximation holds
       when the distance object-camera (d) is much larger than the camera dimensions.
    L_object: float (only required for spherical wave)
        Distance between the pinhole and the object!
    pinhole_center: list of ints (only required for spherical wave)
        Position of the center of the spherical wave, in pixels. If None, we will assume that it corresponds to the
        center of the image

    Returns
    -------
    A list of HPOEImages with the generated hologram at the different requested distances

    """

    # We only need to propagate the field and interfere it with the reference wave

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

    if ref_wave == 'spherical':
        # Need to multiply the aperture field by the spherical reference wave
        med_wavelen = a_field.wav / a_field.n_medium
        r_wave = sph_wave(L_object, pinhole_center, a_field.pixel_spacing, a_field.num_pixels, med_wavelen)
        a_field.im = a_field.im * r_wave

    # Propagate the field at the object plane
    prop_fields = propagate_conv(a_field, d, cfsp, gradient_filter, fresnel_approx)

    # Now interfere the propagated field with the reference wave
    gen_holos = list()

    for i, z_plane_prop_field in enumerate(prop_fields):

        if ref_wave == 'plane':

            r_wave = 1  # plane wave is constant amplitude if it is perpendicular to the propagation axis
            
        elif ref_wave == 'spherical':

            med_wavelen = a_field.wav / a_field.n_medium
            r_wave = sph_wave(L_object + d[i], pinhole_center, a_field.pixel_spacing, a_field.num_pixels, med_wavelen)
            
        holo = z_plane_prop_field
        holo.im = np.abs((holo.im + r_wave)*np.conj(holo.im+r_wave))
        gen_holos.append(holo)

    return gen_holos

# Using the fresnel approach
def generate_hologram_fresnel(aperture_field, d, ref_wave='plane', 
              L_object=None, pinhole_center=None):

    """
    Simulates a hologram for a field at the aperture given by aperture_field, recorded a distance d
    from the object. Uses the fresnel approach to calculate diffraction.

    Remember: holopoe is agnostic to units, and the propagation result will be
    correct as long as the distance and wavelength and pixel spacing are in the same units.

    Parameters
    ----------
    aperture_field : HPOEImage
       The field at the aperture (the scattered field at the object plane)
    d : float or list of floats
       Distance(s) at which the hologram is recorded.  A list tells to
       propagate to several distances and returns a list of HPOEImages at each z.
    ref_wave: string, either 'plane' or 'spherical'
        The type of reference wave that was used to record the hologram.
        If 'spherical', L and pinhole_center need to be provided.
    L_object: float (only required for spherical wave)
        Distance between the pinhole and the object!
    pinhole_center: list of ints (only required for spherical wave)
        Position of the center of the spherical wave, in pixels. If None, we will assume that it corresponds to the
        center of the image

    Returns
    -------
    A list of HPOEImages with the generated hologram at the different requested distances

    """

    # We only need to propagate the field and interfere it with the reference wave

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

    if ref_wave == 'spherical':
        # Need to multiply the aperture field by the spherical reference wave
        med_wavelen = a_field.wav / a_field.n_medium
        r_wave = sph_wave(L_object, pinhole_center, a_field.pixel_spacing, a_field.num_pixels, med_wavelen)
        a_field.im = a_field.im * r_wave

    # Propagate the field at the object plane
    prop_fields = propagate_fresnel(a_field, d)

    # Now interfere the propagated field with the reference wave
    gen_holos = list()

    for i, z_plane_prop_field in enumerate(prop_fields):

        if ref_wave == 'plane':

            r_wave = 1  # plane wave is constant amplitude if it is perpendicular to the propagation axis
            
        elif ref_wave == 'spherical':

            med_wavelen = a_field.wav / a_field.n_medium
            r_wave = sph_wave(L_object + d[i], pinhole_center, a_field.pixel_spacing, a_field.num_pixels, med_wavelen)
            
        holo = z_plane_prop_field
        holo.im = np.abs((holo.im + r_wave)*np.conj(holo.im+r_wave))
        gen_holos.append(holo)

    return gen_holos