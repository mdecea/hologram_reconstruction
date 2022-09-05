# Applies the twin image elimination methods described in 
# [1] Mudanyali, ..., Ozcan: 
# "Compact, light-weight and cost-effective microscope based on lenslessincoherent holography for telemedicine applications",
# Lab on a Chip 2010

import copy
from tqdm import tqdm
from backpropagate.convolution_approach import backpropagate_conv
import numpy as np
from skimage import restoration, util, filters

def interf_phase_retrieval_Ozcan(hologram, support_mask, d, num_iters, beta = 2):

    """
    Applies iterative interferometric phase retrieval (described in [1]) to eliminate the twin image. 
    Assumes a plane wave incident wave, but it is also valid for holograms recorded with spherical waves 
    as long as the reconstruction distance is corrected.
    
    Parameters
    ----------
    hologram : HPOEImage
       The recorded hologram.
    support_mask: 2D numpy array
        A mask with '0' in the pixels where there is an object. Usually this mask is obtained through
        computational_techniques.detect_object_supports methods.
    d : float
       Reconstruction distance.
    num_iters: int
        Number of iterations to apply       
    beta: float
        relaxation parameter. Usually between 2 and 3. A larger value speeds up convergence 
        but is more sensitive to bacground noise. 
    """

    # Step (1): Get image at the sample plane
    u_rec = backpropagate_conv(hologram, d, ref_wave='plane', DC_suppress=False, filt_kernel= None)[0]
    
    u_z2 = copy.deepcopy(u_rec)

    # Step (2): Substitute regions inside the supports by the average value of u_rec
    inds = np.where(support_mask == 0)
    u_z2.im[inds] = np.average(u_z2.im[inds])

    for i in tqdm(range(num_iters)):

        # Step (3): propagate to virtual image plane
        u_mz2 = backpropagate_conv(u_z2, -2*d, ref_wave='plane', DC_suppress=False, filt_kernel= None)[0]

        # Step (4): Slowly set region outside supports to a DC value
        inds = np.where(support_mask == 1)
        bg_avg = np.sum(u_mz2.im[inds])/len(inds[0])
        u_mz2.im[inds] = bg_avg - (bg_avg - u_mz2.im[inds])/beta 

        # Step (5): go back to the image plane
        u_z2 = backpropagate_conv(u_mz2, 2*d, ref_wave='plane', DC_suppress=False, filt_kernel= None)[0]
        u_z2.im[inds] = u_rec.im[inds]
    
    return u_mz2

def non_interf_phase_retrieval_Ozcan(hologram, support_mask, d, num_iters, background=None, obj_color='dark'):

    """
    Applies non-iterative interferometric phase retrieval (described in [1]) to eliminate the twin image. 
    Assumes a plane wave incident wave, but it is also valid for holograms recorded with spherical waves 
    as long as the reconstruction distance is corrected.
    
    Parameters
    ----------
    hologram : HPOEImage
       The recorded hologram.
    support_mask: 2D numpy array
        A mask with '0' in the pixels where there is an object. Usually this mask is obtained through
        computational_techniques.detect_object_supports methods.
    d : float
       Reconstruction distance.
    num_iters: int
        Number of iterations to apply  
    background: HPOEImage or None
        This method requires the background image to be recorded at the camera plane. 
        If set to None, we get the background using the scikit rolling ball algorithm.
    obj_color: string, either 'dark' or 'light'
        This is only relevant in case background is None and is used by the rolling ball
        algorithm. If obj_color is 'dark', we consider that the background is light and objects dark.
        If obj_color is 'light', then we assume the opposite.
    """

    # Step (1a): Get square root of hologram
    h_rec = copy.deepcopy(hologram)
    h_rec.im = np.sqrt(h_rec.im)

    # Step (1b): Get background and propagate it to the virtual image plane
    if background is None:
        background = get_background(hologram, obj_color)
    
    d_mz2 = backpropagate_conv(background, -d, ref_wave='plane', DC_suppress=False, filt_kernel= None)[0]

    u_camera = copy.deepcopy(h_rec)

    for i in tqdm(range(num_iters)):

        # Step (2): propagate to virtual image plane
        u_mz2 = backpropagate_conv(u_camera, -d, ref_wave='plane', DC_suppress=False, filt_kernel= None)[0]

        # Step (3): Substitute values outside supports by a weighted average between the background and the propagated holo
        inds = np.where(support_mask == 1)
        m = np.average(u_mz2.im)/np.average(u_mz2.im)
        u_mz2.im[inds] = m * d_mz2.im[inds]

        # Step (4): propagate back to camera plane
        u_camera = backpropagate_conv(u_mz2, d, ref_wave='plane', DC_suppress=False, filt_kernel= None)[0]

        # Step (5): keep the phase of the obtained field at the camera, use the original intensity
        phase = np.angle(u_camera.im)
        u_camera.im = h_rec.im * np.exp(1j * phase)
    
    return u_mz2

def get_background(holo, obj_color, sigma=50):
    """
    Gets the hologram background by using the rolling ball algorithm from scikit
    https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rolling_ball.html
    """

    bg = copy.deepcopy(holo)


    ## The algorithm assumes that the background is dark. If not, we need to invert the image.
    #if obj_color == 'light':
    #    background = restoration.rolling_ball(holo.im)
    #else:
    #    # Invert the image
    #    background = util.invert(restoration.rolling_ball(util.invert(holo.im)))
    
    
    background = filters.gaussian(
        holo.im, sigma=sigma, multichannel=False)

    bg.im = background
    
    return bg

