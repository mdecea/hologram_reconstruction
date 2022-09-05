'''
Example script for sample reconstruction from an acquired hologram.
The example file corresponds to a real hologram acquired on 20 um diameter beads
illuminated by an 810 nm LED.
'''

import sys
sys.path.insert(0, "../")

import matplotlib.pyplot as plt
from image import HPOEImage
from inout.visualize import plot_image
import process.image_processing
from backpropagate.convolution_approach import backpropagate_conv
from inout.io import save_image

plt.rcParams['figure.figsize'] = [12, 9]
plt.rcParams['figure.dpi'] = 100

# -----------------------------------------------------
# LOADING AND PROCESSING

# Load the hologram and the background
# We used a ZWOASI1600MM camera to record the hologram. This cmaeras has 3.8 um pixel spacing.
holo = HPOEImage(path='20um_beads_holo_Ibias=30.5mA.png',
                 image=None, pixel_spacing=3.8, n_medium=1.0, wav=0.81, channels=0)
# The background is loaded from an average
bg = HPOEImage.from_average('./', pixel_spacing=3.8, n_medium=1.0,
                            wav=0.81, channels=0, image_glob='20um_beads_bg*.png')

# Subtract background from the hologram, plot it and save it
holo_sub = process.image_processing.bg_correct(holo, bg, dark_image = None, operation = 'divide')
plot_image(holo_sub, axis_units = 'pixel', scaling=None, title = "background subtracted")
save_image('hologram_w_bg_subtraction.png', holo_sub)

# ----------------------------------------------------------
# ----------------------------------------------------------

print('starting backprop')

d = [64.44e3]  # This is the distance that makes the beads be in focus. d can be an array with multiple distances.

# Do the backpropagation
prop_holo_sub = backpropagate_conv(holo_sub, d, ref_wave='plane', DC_suppress=False, filt_kernel=None,
 cfsp=3, gradient_filter=False, fresnel_approx=False )

# Plot the resulting reconstructed mage and save it to a png file
for i, h in enumerate(prop_holo_sub):
    plot_image(h, title='Plane wave reconstruction, d = %.2f mm' % (d[i]*1e-3), mode='intensity', scaling=None)

save_image('hologram_reconstruction.png', h)
plt.show(block=True)