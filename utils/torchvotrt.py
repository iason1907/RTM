import torch.nn
from torch import Tensor
import torchvision.transforms.functional as F

import io
import imageio
import numpy as np
import numbers
from collections.abc import Sequence
import torchvision.transforms.v2.functional as tv2f

class JpegQuality(torch.nn.Module):
    """
    transforms an image tensor simulating jpeg compression artifacts
    param q: quality of compression 0 to 100
    """

    def __init__(self, q=1):
        super().__init__()
        self.q = q

    @staticmethod
    def createArifacts(im, q):
        buf = io.BytesIO()
        imageio.imwrite(buf, im, format='jpg', quality=q)
        s = buf.getbuffer()
        return imageio.imread(s, format='jpg')

    def forward(self, img: Tensor) -> Tensor:
        if img.is_cuda:
            img.cpu()
        img_ = img.type(torch.uint8)
        jpg_im = tv2f.jpeg(img_, quality=self.q)
        if img.is_cuda:
            jpg_im = jpg_im.cuda()
        return jpg_im.float()

class SNP(torch.nn.Module):
    """
    Amount specifies how many pixels will be affected.
    Ratio specifies the ratio of salt to pepper.
    Can be only salt if ratio = 1 or only pepper if ratio = 0.
    """
    def __init__(self, amount=0.25, ratio=0.25):
        super().__init__()
        self.p = amount
        self.q = ratio

    @staticmethod
    def _bernoulli(p, shape, *, random_state):
        """
        Bernoulli trials at a given probability of a given size.
        This function is meant as a lower-memory alternative to calls such as
        `np.random.choice([True, False], size=image.shape, p=[p, 1-p])`.
        While `np.random.choice` can handle many classes, for the 2-class case
        (Bernoulli trials), this function is much more efficient.
        Parameters
        ----------
        p : float
            The probability that any given trial returns `True`.
        shape : int or tuple of ints
            The shape of the ndarray to return.
        Returns
        -------
        out : nd array[bool]
            The results of Bernoulli trials in the given `size` where success
            occurs with probability `p`.
        """
        if p == 0:
            return np.zeros(shape, dtype=bool)
        if p == 1:
            return np.ones(shape, dtype=bool)
        return random_state.random(shape) <= p

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be distorted.
        Returns:
            PIL Image or Tensor: Image with salt and pepper added.
        """
        rng = np.random.default_rng()
        flipped = self._bernoulli(self.p, img.shape, random_state=rng)
        salted = self._bernoulli(self.q, img.shape, random_state=rng)
        peppered = ~salted
        a = [flipped & salted]
        b = [flipped & peppered]
        img.data.numpy()[a[0]] = 1
        img.data.numpy()[b[0]] = 1
        return img


class Poisson(torch.nn.Module):
    def __init__(self, seed=1):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def forward(self, img: Tensor) -> Tensor:
        max_v = img.data.numpy().max()
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        return torch.clip(torch.tensor(self.rng.poisson(img * vals) / float(vals)), 0, max_v).type(torch.FloatTensor)


class WGN(torch.nn.Module):
    """Adds white gaussian noise to image.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        mean

    Returns:
        PIL Image or Tensor: The input image with WGN added.
    """

    def __init__(self, amount=0.5, mean=0, var=1000, seed=1):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.mean = mean
        self.var = var
        self.seed = seed
        self.amount = amount

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be distorted.
        Returns:
            PIL Image or Tensor: Image with WGN added.
        """
        if img.is_cuda:
            # max_v = img.data.cpu().numpy().max()
            max_v = img.max()
            # img = img/max_v
            noise = ((self.var**0.5)*torch.randn(img.shape)).cuda()
            # noise = torch.from_numpy(self.rng.normal(self.mean,
            #                         self.var ** 0.5, img.shape)).cuda()
            return torch.clip((img +  noise), 0, max_v).float()
        else:
            max_v = img.data.numpy().max()
            noise = self.rng.normal(self.mean, self.var ** 0.5, img.shape)
            return torch.clip((img +  noise), 0, max_v).float()


class GaussianBlur(torch.nn.Module):
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = self._setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def _setup_size(size, error_msg):
        if isinstance(size, numbers.Number):
            return int(size), int(size)

        if isinstance(size, Sequence) and len(size) == 1:
            return size[0], size[0]

        if len(size) != 2:
            raise ValueError(error_msg)

        return size

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        sigma = self.get_params(self.sigma[0], self.sigma[1])
        return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])


class Speckle(torch.nn.Module):
    """Adds speckle noise to image.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        mean

    Returns:
        PIL Image or Tensor: The input image with WGN added.
    """

    def __init__(self, mean=0, var=1, seed=1):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.mean = mean
        self.var = var
        self.seed = seed

    def forward(self, img: Tensor) -> Tensor:
        """
        Args:
            img (Tensor): image to be distorted.
        Returns:
            Tensor: Image with Speckle added.
        """
        if img.is_cuda:
            noise = ((self.var**0.5)*torch.randn(img.shape)).cuda()
        else:
            noise = ((self.var**0.5)*torch.randn(img.shape))
        return img + img * noise

# transformations
transform_q = torch.nn.Sequential(
    JpegQuality(q=7)
)

transform_blur = torch.nn.Sequential(
    GaussianBlur(5, sigma=(0.1, 2.0)),
)

transform_wgn = torch.nn.Sequential(
    WGN()
)

transform_snp = torch.nn.Sequential(
    SNP(amount=0.25, ratio=0.25)
)

transform_poisson = torch.nn.Sequential(
    Poisson()
)

transform_speckle = torch.nn.Sequential(
    Speckle(mean=0, var=900)
)


def noise_handler(img, noise_type):
    if noise_type == 'original':
        return img
    elif noise_type == 'WGN':
        return transform_wgn(torch.tensor(img)).data.numpy().astype(np.uint8)
    elif noise_type == 'SnP':
        return transform_snp(torch.tensor(img)).data.numpy().astype(np.uint8)
    else:
        return transform_blur(torch.tensor(img.transpose(2,0,1))).data.numpy().transpose(1,2,0).astype(np.uint8).copy()

def noise_handler_tensor(img, noise_type):
    if noise_type == 'original':
        return img
    elif noise_type == 'WGN':
        return transform_wgn(img)
    elif noise_type == 'SnP':
        return transform_snp(img)
    else:
        return transform_blur(img)