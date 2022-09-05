
from backpropagate.convolution_approach import backpropagate_conv
import copy
import numpy as np
from scipy.ndimage import convolve
from tqdm import tqdm
from inout.visualize import plot_image


# Applies phase retrieval algorithm described in 
# [1] Latychevskaia and Fink: 
# "Reconstruction of purely absorbing, absorbing and phase-shifting, and strong phase-shifting objects from their single-shot in-line holograms"


def iterative_reconstruction_Fink(hologram, d, num_iters, support_mask=None):

    """
    Applies iterative reconstruction (described in [1]) to reconstruct the hologram. Assumes a plane wave incident wave,
    but it is also valid for holograms recorded with spherical waves as long as the reconstruction distance is corrected.
    
    Parameters
    ----------
    hologram : HPOEImage
       The recorded hologram, with the background corrected through division (not subtraction nor divide_and_subtract).
    d : float
       Reconstruction distance.
    num_iters: int
        Number of iterations to apply
    support_mask: 2D numpy array or None
        If provided, this is a mask with '0' in the pixels where there is an object. Usually this mask is obtained through
        computational_techniques.detect_object_supports methods. It can hel speed up convergence of the iterative method.
        
    """

    original_hologram = copy.deepcopy(hologram)

    camera_plane_wavefront = copy.deepcopy(hologram)
    camera_plane_wavefront.im = np.sqrt(camera_plane_wavefront.im)

    for i in tqdm(range(num_iters)):

        # Step (1): Propagate hologram to the sample plane. 
        # We assume any DC suppression has been already applied.
        sample_plane_wf = backpropagate_conv(camera_plane_wavefront, d, ref_wave='plane', DC_suppress=False, filt_kernel= None)
        sample_plane_wf = sample_plane_wf[0]
        #plot_image(sample_plane_wf, title='non-mod', mode='intensity', scaling = [0, 1.5])

        # Step (2a): If the support_mask is provided, use it.
        # We set sample_plane_wf to 1 where there is no object (i.e, support_mask = 1), and do not modify the rest.
        if support_mask is not None:
            inds = np.where(support_mask == 1)
            sample_plane_wf.im[inds] = 1 

        # Step (2b): Substitute any values that have negative absorption (i.e, gain)
        abs_val = np.abs(sample_plane_wf.im)
        indices = np.where( (-1 * np.log(abs_val)) < 0)
        print(len(indices[0]))
        sample_plane_wf.im[indices] = 1
        #plot_image(sample_plane_wf, title='mod', mode='intensity', scaling = [0, 1.5])

        # Step (3): Apply a smoothing filter - skip for now
        # filt_kernel = (1/25)*np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 4, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        filt_kernel = (1/9)*np.array([ [ 1, 1, 1], [1, 4, 1], [1, 1, 1]])
        for dim in range(sample_plane_wf.im.shape[2]):
            relevant_array = sample_plane_wf.im[:,:,dim]
            abs_val = np.abs(relevant_array)
            phase_val = np.angle(relevant_array)
            abs_val_smoothed = convolve(abs_val, filt_kernel)
            phase_val_smoothed = convolve(phase_val, filt_kernel)
            sample_plane_wf.im[:,:,dim] = abs_val_smoothed*np.exp(1j*phase_val_smoothed)

        # Step (4): Propagate back to the camera plane
        camera_plane_wavefront = backpropagate_conv(sample_plane_wf, -d, ref_wave='plane', DC_suppress=False, filt_kernel= None)
        camera_plane_wavefront = camera_plane_wavefront[0]

        # Step (5): The updated phase distribution is that obatined in step (4), the intensity is the one given by the orginal hologram
        camera_plane_wavefront.im = np.sqrt(original_hologram.im) * np.exp(1j * np.angle(camera_plane_wavefront.im))

        #plot_image(camera_plane_wavefront, title='phase at camera plane', mode='phase', scaling = 'auto')
        #input()

    return sample_plane_wf