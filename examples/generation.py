'''
Example script for hologram generation from a mask showing the object at the sample plane.
'''

import sys
sys.path.insert(0, "../")

import numpy as np
import matplotlib.pyplot as plt
from image import HPOEImage
from process.image_processing import get_line
from inout.visualize import plot_image
from generate.generate_hologram import generate_hologram_conv
from process.image_processing import convert_to_grayscale, pad_image


# For fun, we will generate a hologram at different wavelengths and sum them up 
# into a composite hologram. This could be used, for example, to see the effects of using
# a broadband source (low temporal coherence) as illumination.

# We will simulate the hologram at all these wavelengths
wavs = np.linspace(0.81-0.05, 0.81+0.05, 21)

plot_ind_holos = False # If True it will plot all the holograms at each wavelength (will generate 21 plots)

# We will simulate the holograms at all these propagation distances
d = [1e4] 

######## LOAD DATA AND SHOW IT #########

# Load the "mask" showing the object being illuminated.
aperture_field = HPOEImage(path='./bead.png', 
                 image=None, pixel_spacing=1, n_medium=1.0, wav=0.81, channels=[0, 1, 2])

aperture_field = convert_to_grayscale(aperture_field, method='average')
aperture_field = pad_image(aperture_field, [2048, 2048])

plot_image(aperture_field, axis_units = 'pixel', scaling=None, title = "Field at the aperture")

# This is the container for the composite hologram (the sum of all the holograms
# at the different wavelengths)
composite_holo = HPOEImage(path=None, image=np.zeros_like(aperture_field.im), pixel_spacing=5,
 n_medium=1.0, wav=0.81, channels=None)

line_profiles = list()

for wav in wavs:

    ####### GENERATE THE HOLOGRAMS ##########
    aperture_field.wav = wav

    # Plane wave reference
    gen_holos = generate_hologram_conv(aperture_field, d, ref_wave='plane', cfsp=0,
                gradient_filter=False, fresnel_approx=False,
                L_object=None, pinhole_center=None)

    for i, h in enumerate(gen_holos):

        if wav == wavs[0]:
            plot_image(h, title = 'Gen holo, plane wave, convolution, d = %.2f mm, lam = %.2f um' % (d[i]*1e-3, wav) )
        if plot_ind_holos:
            plot_image(h, title = 'Gen holo, plane wave, convolution, d = %.2f mm, lam = %.2f um' % (d[i]*1e-3, wav) )
        line_profiles.append(get_line(h, 'x', interactive=False, extent=300, coord=[1088, 1088]))

        composite_holo = composite_holo + h

plot_image(composite_holo, title = 'Composite holo, plane wave, d = %.2f mm' % (d[-1]*1e-3) )

plt.figure()
tot_profile = np.zeros_like(line_profiles[0])

for prof in line_profiles:
    plt.plot(prof/np.max(prof))
    tot_profile = tot_profile + prof

plt.figure()
plt.plot(tot_profile/np.max(tot_profile))

plt.show(block=True)