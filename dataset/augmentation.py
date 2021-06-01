import numpy as np
import numpy.random as random

from PIL import Image, ImageFilter

import skimage
from skimage import color


# from patches_dataset & bacteria_bag
# MEAN = [218.78237903, 194.69390888, 201.95128081]
# STD = [2.61339548, 9.74849698, 4.84232922]


class RandomVerticalFlip(object):
    """Vertically flip the given PIL.Image randomly with a probability of 0.5."""
    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotate(object):
    """Rotate the given PIL.Image by either 0, 90, 180, 270."""
    def __call__(self, img):
        return img.rotate(random.choice([0, 90, 180, 270]))


class RandomHEStain(object):
    """Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """
    def __call__(self, img):
        img_he = skimage.color.rgb2hed(img)
        img_he[:, :, 0] = img_he[:, :, 0] * random.normal(1.0, 0.02, 1)  # H
        img_he[:, :, 1] = img_he[:, :, 1] * random.normal(1.0, 0.02, 1)  # E
        img_rgb = np.clip(skimage.color.hed2rgb(img_he), 0, 1)
        img = Image.fromarray(np.uint8(img_rgb*255.999))
        return img


class RandomGaussianNoise(object):
    """Transfer the given PIL.Image from rgb to HE, perturbate, transfer back to rgb """
    def __call__(self, img):
        img = img.filter(ImageFilter.GaussianBlur(random.normal(0.0, 0.5, 1)))
        return img


class HistoNormalize(object):
    """Normalizes the given PIL.Image"""
    def normalize(image, target=np.array([[57.4, 15.84], [39.9, 9.14], [-22.34, 6.58]])):
        """Normalizing function we got from the cedars-sinai medical center"""

        whitemask = color.rgb2gray(image)
        whitemask = whitemask > (215 / 255)

        imagelab = color.rgb2lab(image)

        imageL, imageA, imageB = [imagelab[:, :, i] for i in range(3)]

        # mask is valid when true
        imageLM = np.ma.MaskedArray(imageL, whitemask)
        imageAM = np.ma.MaskedArray(imageA, whitemask)
        imageBM = np.ma.MaskedArray(imageB, whitemask)

        ## Sometimes STD is near 0, or 0; add epsilon to avoid div by 0 -NI
        epsilon = 1e-11

        imageLMean = imageLM.mean()
        imageLSTD = imageLM.std() + epsilon

        imageAMean = imageAM.mean()
        imageASTD = imageAM.std() + epsilon

        imageBMean = imageBM.mean()
        imageBSTD = imageBM.std() + epsilon

        # normalization in lab
        imageL = (imageL - imageLMean) / imageLSTD * target[0][1] + target[0][0]
        imageA = (imageA - imageAMean) / imageASTD * target[1][1] + target[1][0]
        imageB = (imageB - imageBMean) / imageBSTD * target[2][1] + target[2][0]

        imagelab = np.zeros(image.shape)
        imagelab[:, :, 0] = imageL
        imagelab[:, :, 1] = imageA
        imagelab[:, :, 2] = imageB

        # Back to RGB space
        returnimage = color.lab2rgb(imagelab)
        returnimage = np.clip(returnimage, 0, 1)
        returnimage *= 255
        # Replace white pixels
        returnimage[whitemask] = image[whitemask]
        return returnimage.astype(np.uint8)

    def __call__(self, img):
        img_arr = np.array(img)
        img_norm = self.normalize(img_arr)
        img = Image.fromarray(img_norm)
        return img
