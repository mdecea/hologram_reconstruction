import warnings
import numpy as np
from holopoe.errors import BadImage
from holopoe.inout.io import load_image, load_average


class HPOEImage(object):

    """Main class of holopoe, which holds an image (which can be a hologram or a hologram
    reconstruction) as a numpy array andsome metssadata.

    Units of distances, pixel_spacing and wavelength don't matter as long as
    they are consistent.

    Two initialization methods:
    1. Providing the numpy array image directly.
    2. Providing a path to an image and the channels to load.

    If the parameter 'image' is not None, we will assume case 1 above. If 'image' is None,
    then we will try to load the image specified by the 'path' and 'channels' parameters.

    We can also initialize a hologram from an average, which can be useful when
    we have background data. To do so, we can call the factory
    holo = Image.from_average(path, global). Check the function for more details.

    Parameters
    ----------
    path : string (optional)
        path to the hologram or image
    image: numpy array (optional)
        A 3D numpy array with the image. Dim 1 is x, Dim 2 is y, Dim 3 is color (or channel).
    pixel_spacing : float or (float, float) (optional)
        pixel pitch of camera in each dimension - assumes square pixels if single value.
        set equal to 1 if not passed in and issues warning.
    n_medium : float (optional)
        refractive index of the medium across which the light propagates
    wav : float (optional)
        wavelength (in vacuum) of illuminating light.
    channels : int or tuple of ints or 'all' (optional)
        number(s) of channel to load for a color image (in general 0=red,
        1=green, 2=blue)
    """

    def __init__(self, path=None, image=None, pixel_spacing=None, n_medium=1.0, wav=None, channels='all'):

        # First, differentiate between case 1 and 2 to construct the image
        if image is not None:
            # We are in case 1 above --> The image is provided to us
            self.im = image
        else:
            # We need to load the image from the path
            self.im = load_image(path=path, channels=channels)

        # Common operations
        if len(self.im.shape) != 3:
            raise BadImage("The provided image array does not have 3 dimensions.")

        self.num_channels = self.im.shape[2]  # 3rd dimension of the array is number of channels
        self.num_pixels = self.im.shape[0:2]

        if not isinstance(pixel_spacing, list):
            # Only one spacing provided --> We assume it is the same in x and y
            self.pixel_spacing = [pixel_spacing, pixel_spacing]
        else:
            if len(pixel_spacing) == 1:
                self.pixel_spacing = [pixel_spacing[0], pixel_spacing[0]]
            else:
                self.pixel_spacing = pixel_spacing[0:2]

        self.n_medium = n_medium
        self.wav = wav


    @classmethod
    def from_average(cls, path, pixel_spacing=None, n_medium=1.0, wav=None, channels='all', image_glob='*.tif'):
        """
        Parameters
        ----------
        path: string or list(string)
            Directory or list of filenames or filepaths. If filename is a
            directory, it will average all images matching image_glob.
        pixel_spacing : float or (float, float) (optional)
            pixel pitch of camera in each dimension - assumes square pixels if single value.
            set equal to 1 if not passed in and issues warning.
        n_medium : float (optional)
            refractive index of the medium across which the light propagates
        wav : float (optional)
            wavelength (in vacuum) of illuminating light. 
        channels : int or tuple of ints or 'all' (optional)
            number(s) of channel to load for a color image (in general 0=red,
            1=green, 2=blue)
        image_glob : string
            Glob used to select images (if images is a directory)
        """

        average_image = load_average(path, channels, image_glob)
        return cls(path=None, image=average_image, pixel_spacing=pixel_spacing, n_medium=n_medium,
         wav=wav, channels=None)

    def get_metadata(self):
        """
        Returns the Image metadata
        """
        return self.pixel_spacing, self.n_medium, self.wav

    def __str__(self):
        """
        Prints metadata.
        """
        metadata = ""

        if self.pixel_spacing is not None:
            metadata = metadata + ("Pixel spacing: [%.2f, %.2f] ; " % (self.pixel_spacing[0], self.pixel_spacing[1]))
        if self.wav is not None:
            metadata = metadata + ("Wavelength: %.3f ; " % self.wav)
        if self.n_medium is not None:
            metadata = metadata + ("Medium refractive index: %.2f ; " % self.n_medium)
        if self.num_pixels is not None:
            metadata = metadata + ("Num pixels: [%.2f, %.2f] ; " % (self.num_pixels[0], self.num_pixels[1]))
        
        return metadata

    def __sub__(self, other):
        """
        Define subtraction of an HPOEImage with either a constant value or another HPOEImage.
        """

        p_sp, n, wav = self.get_metadata()

        if isinstance(other, HPOEImage):
            p_sp1, n1, wav1 = other.get_metadata()

            if p_sp != p_sp1:
                warnings.warn("The two HPOEImages to subtract don't have the same pixel spacing.")
            if n != n1:
                warnings.warn("The two HPOEImages to subtract weren't taken in a medium with the same refractive index.")
            if wav != wav1:
                warnings.warn("The two HPOEImages to subtract weren't taken at the same wavelength.")

            subtr_im = self.im - other.im
        else:
            subtr_im = self.im - other

        return HPOEImage(path=None, image=subtr_im, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
    
    def __add__(self, other):
        """
        Define addition of an HPOEImage with either a constant value or another HPOEImage.
        """

        p_sp, n, wav = self.get_metadata()

        if isinstance(other, HPOEImage):
            p_sp1, n1, wav1 = other.get_metadata()

            if p_sp != p_sp1:
                warnings.warn("The two HPOEImages to add don't have the same pixel spacing.")
            if n != n1:
                warnings.warn("The two HPOEImages to add weren't taken in a medium with the same refractive index.")
            if wav != wav1:
                warnings.warn("The two HPOEImages to add weren't taken at the same wavelength.")

            subtr_im = self.im + other.im
        else:
            subtr_im = self.im + other

        return HPOEImage(path=None, image=subtr_im, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
    
    def __mult__(self, other):
        """
        Define multiplication of an HPOEImage with either a constant value or another HPOEImage.
        It does elementwise multiplication.
        """

        p_sp, n, wav = self.get_metadata()

        if isinstance(other, HPOEImage):
            p_sp1, n1, wav1 = other.get_metadata()

            if p_sp != p_sp1:
                warnings.warn("The two HPOEImages to multiply don't have the same pixel spacing.")
            if n != n1:
                warnings.warn("The two HPOEImages to multiply weren't taken in a medium with the same refractive index.")
            if wav != wav1:
                warnings.warn("The two HPOEImages to multiply weren't taken at the same wavelength.")

            subtr_im = np.multiply(self.im, other.im)
        else:
            subtr_im = self.im * other

        return HPOEImage(path=None, image=subtr_im, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)

    def __truediv__(self, other):
        """
        Define division of an HPOEImage with either a constant value or another HPOEImage.
        It does elementwise division.
        """

        p_sp, n, wav = self.get_metadata()

        if isinstance(other, HPOEImage):
            p_sp1, n1, wav1 = other.get_metadata()

            if p_sp != p_sp1:
                warnings.warn("The two HPOEImages to divide don't have the same pixel spacing.")
            if n != n1:
                warnings.warn("The two HPOEImages to divide weren't taken in a medium with the same refractive index.")
            if wav != wav1:
                warnings.warn("The two HPOEImages to divide weren't taken at the same wavelength.")

            subtr_im = np.divide(self.im, other.im)
        else:
            subtr_im = self.im / other

        return HPOEImage(path=None, image=subtr_im, pixel_spacing=p_sp, n_medium=n, wav=wav, channels=None)
