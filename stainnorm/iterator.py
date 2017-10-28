from __future__ import absolute_import

""" Data augmentation routines

Data augmentation routines

Note
-----
Mainly inspired by the implementation of the image preprocessor in Keras (https://keras.io/)

Taken from my bachelor's thesis code, originally named ``ilu_deepml.image_pre`` with some more
additional functions that were omitted for the purpose of this project.
"""

__author__ = "sschneider"
__email__ = "steffen.schneider@rwth-aachen.de"

import os
import logging, re

import numpy as np
from scipy import ndimage, linalg

from os import listdir
from os.path import isfile, join
import random, math
from six.moves import range

from skimage.color import rgb2hsv, hsv2rgb
from sklearn import preprocessing

cwh2whc = (1, 2, 0)
whc2cwh = (2, 0, 1)

datagen = None
warn_once = False

'''
    Fairly basic set of tools for realtime data augmentation on image data.
    Can easily be extended to include new transforms, new preprocessing methods, etc...
'''

def hsv_augmentation(x, sigma, form_inp='rgb', form_outp='rgb'):
    global warn_once
    assert len(sigma) == 3
    log = logging.getLogger(__name__)

    rescale = False
    if not ( x.max() <= 1.001 and x.min() >= 0 ):
        rescale = True
        if not warn_once:
            log.warning("Input range not in 0...1. Assuming 0...255 as input range!") 
            warn_once = True
        x /= 255
    x = np.clip(x, 0, 1)
    if form_inp == 'rgb':
        x = rgb2hsv(x.transpose(cwh2whc))
    x[:,:,0] = (x[:,:,0] +  np.random.normal(0., sigma[0])) % 1
    x[:,:,1] = np.clip((x[:,:,1] +  np.random.normal(0., sigma[1])), 0, 1)
    x[:,:,2] = np.clip((x[:,:,2] +  np.random.normal(0., sigma[2])), 0, 1)
    if form_outp == 'rgb':
        x = hsv2rgb(x).transpose(whc2cwh)

    if rescale:
        x *= 255
    return x 

def random_rotation(x, rg, fill_mode="nearest", cval=0.):
    angle = random.uniform(-rg, rg)
    x = ndimage.interpolation.rotate(x, angle, axes=(1,2), reshape=False, mode=fill_mode, cval=cval)
    return x

def random_shift(x, wrg, hrg, fill_mode="nearest", cval=0.):
    crop_left_pixels = 0
    #crop_right_pixels = 0
    crop_top_pixels = 0
    #crop_bottom_pixels = 0

    #original_w = x.shape[1]
    #original_h = x.shape[2]

    if wrg:
        crop = random.uniform(0., wrg)
        split = random.uniform(0, 1)
        crop_left_pixels = int(split*crop*x.shape[1])
        #crop_right_pixels = int((1-split)*crop*x.shape[1])

    if hrg:
        crop = random.uniform(0., hrg)
        split = random.uniform(0, 1)
        crop_top_pixels = int(split*crop*x.shape[2])
        #crop_bottom_pixels = int((1-split)*crop*x.shape[2])

    x = ndimage.interpolation.shift(x, (0, crop_left_pixels, crop_top_pixels), mode=fill_mode, cval=cval)
    return x

def horizontal_flip(x):
    for i in range(x.shape[0]):
        x[i] = np.fliplr(x[i])
    return x

def vertical_flip(x):
    for i in range(x.shape[0]):
        x[i] = np.flipud(x[i])
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass

def random_shear(x, intensity):
    # TODO
    pass

def random_channel_shift(x, rg):
    # TODO
    pass

def random_zoom(x, rg, fill_mode="nearest", cval=0.):
    zoom_w = random.uniform(1.-rg, 1.)
    zoom_h = random.uniform(1.-rg, 1.)
    x = ndimage.interpolation.zoom(x, zoom=(1., zoom_w, zoom_h), mode=fill_mode, cval=cval)
    return x # shape of result will be different from shape of input!

def array_to_img(x, scale=True):
    from PIL import Image
    x = x.transpose(1, 2, 0) 
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype("uint8"), "RGB")
    else:
        # grayscale
        return Image.fromarray(x[:,:,0].astype("uint8"), "L")


def img_to_array(img):
    x = np.asarray(img, dtype='float32')
    if len(x.shape)==3:
        # RGB: height, width, channel -> channel, height, width
        x = x.transpose(2, 0, 1)
    else:
        # grayscale: height, width -> channel, height, width
        x = x.reshape((1, x.shape[0], x.shape[1]))
    return x


def load_img(path, grayscale=False):
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else: # Assure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [join(directory,f) for f in listdir(directory) \
        if isfile(join(directory,f)) and re.match('([\w]+\.(?:' + ext + '))', f)]

def standardize(x):
    global datagen
    if datagen.featurewise_center:
        x -= datagen.mean
    if datagen.featurewise_std_normalization:
        x /= datagen.std

    if datagen.zca_whitening:
        flatx = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2]))
        whitex = np.dot(flatx, datagen.principal_components)
        x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

    if datagen.samplewise_center:
        x -= np.mean(x)
    if datagen.samplewise_std_normalization:
        x /= np.std(x)

    return x

def random_transform(x):
    global datagen
    if datagen.rotation_range:
        x = random_rotation(x, datagen.rotation_range)
    if datagen.width_shift_range or datagen.height_shift_range:
        x = random_shift(x, datagen.width_shift_range, datagen.height_shift_range)
    if datagen.horizontal_flip:
        if random.random() < 0.5:
            x = horizontal_flip(x)
    if datagen.vertical_flip:
        if random.random() < 0.5:
            x = vertical_flip(x)
    if datagen.hsv_aug_sigma is not None:
        x = hsv_augmentation(x, datagen.hsv_aug_sigma)
    if datagen.rot90 != 0:
        times = int(np.random.randint(datagen.rot90))
        x = np.rot90(x.transpose(cwh2whc), times).transpose(whc2cwh)


def transf(x):
    x = random_transform(x)
    #x = standardize(x)
    return x

class ImageDataGenerator(object):

    def __init__(self,
            featurewise_center=False, # set input mean to 0 over the dataset
            samplewise_center=False, # set each sample mean to 0
            featurewise_std_normalization=False, # divide inputs by std of the dataset
            samplewise_std_normalization=False, # divide each input by its std
            zca_whitening=False, # apply ZCA whitening
            rotation_range=0., # degrees (0 to 180)
            width_shift_range=0., # fraction of total width
            height_shift_range=0., # fraction of total height
            horizontal_flip=False,
            vertical_flip=False,
            hsv_aug_sigma=None,
            rot90=0.
        ):
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None


    def standardize(self, x):
        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= self.std

        if self.zca_whitening:
            flatx = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2]))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        if self.samplewise_center:
            x -= np.mean(x)
        if self.samplewise_std_normalization:
            x /= np.std(x)

        return x


    def random_transform(self, x):
        if self.rotation_range:
            x = random_rotation(x, self.rotation_range)
        if self.width_shift_range or self.height_shift_range:
            x = random_shift(x, self.width_shift_range, self.height_shift_range)
        if self.horizontal_flip:
            if random.random() < 0.5:
                x = horizontal_flip(x)
        if self.vertical_flip:
            if random.random() < 0.5:
                x = vertical_flip(x)
        if self.hsv_aug_sigma is not None:
            x = hsv_augmentation(x, self.hsv_aug_sigma)
        if self.rot90 != 0:
            times = int(np.random.randint(self.rot90))
            x = np.rot90(x.transpose(cwh2whc), times).transpose(whc2cwh)

        # TODO:
        # zoom
        # barrel/fisheye
        # shearing
        # channel shifting
        return x

def distribute(nb_classes, batch_size):
    # Distributes batch_size over different classes 
    n = np.ones((nb_classes,)) * int(batch_size/nb_classes)
    rest = batch_size - n.sum()   
    if rest > 0:
        x = np.random.random_integers(0, nb_classes-1, size=(rest,))
        n[x] += np.array([np.sum(x==i) for i in x])
    #assert int(n.sum()) == batch_size,  "sum: %s, batch_size: %s" % (str(n.sum()), str(batch_size))
    return tuple(n.astype('int'))

class PatchDataGenerator(ImageDataGenerator):
    '''
        Generate minibatches with 
        realtime data augmentation.
    '''
    def __init__(self, **kwargs): 
        super(PatchDataGenerator, self).__init__(**kwargs)


    def flow(self, X, y, batch_size=32, shuffle=False, seed=None, save_to_dir=None, save_prefix="", save_format="jpeg", unsupervised=False):
        log = logging.getLogger(__name__)
        if seed:
            random.seed(seed)

        if shuffle:
            log.info("Shuffling the dataset")
            seed = random.randint(1, 10e6)
            np.random.seed(seed)
            np.random.shuffle(X)
            np.random.seed(seed)
            np.random.shuffle(y)

        nb_batch = int(math.ceil(float(X.shape[0])/batch_size))
        
        if unsupervised:
            self.horizontal_flip = False
            self.vertical_flip = False
            self.rot90 = 0
        
        for b in range(nb_batch):
            batch_end = (b+1)*batch_size
            if batch_end > X.shape[0]:
                nb_samples = X.shape[0] - b*batch_size
            else:
                nb_samples = batch_size

            bX = np.zeros(tuple([nb_samples]+list(X.shape)[1:]))
            
            self.parallel = False
            for i in range(nb_samples):
                x = X[b*batch_size+i]
                x = self.random_transform(x.astype("float32"))
                x = self.standardize(x)
                bX[i] = x
            
            if unsupervised:
                yield bX, X[b*batch_size:b*batch_size+nb_samples,:]
            else:
                yield bX, y[b*batch_size:b*batch_size+nb_samples,:]


    def fit(self, X,
            augment=False,  # fit on randomly augmented samples
            rounds=1,  # if augment, how many augmentation passes over the data do we use
            seed=None):
        '''
            Required for featurewise_center, featurewise_std_normalization and zca_whitening.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds*X.shape[0]]+list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    img = array_to_img(X[i])
                    img = self.random_transform(img)
                    aX[i+r*X.shape[0]] = img_to_array(img)
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean
        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= self.std

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
            fudge = 10e-6
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + fudge))), U.T)

'''
Helper functions for quick prototyping
'''

def get_datagen_ae():
    return PatchDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                zca_whitening=False,
                horizontal_flip=False,
                vertical_flip=False,
                hsv_aug_sigma=(0.1, 0.1, 0.1),
                rot90=0
                )

def get_datagen(mode, augment=True, model_name=""):
    '''
    Contstruct data generator with standard parameters
    '''
    if "autoencoder" in model_name or "unet" in model_name:
        return get_datagen_ae()
    assert mode in ["slide", "patch"]
    if mode is "patch":
        if augment:
            datagen = PatchDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                zca_whitening=False,
                horizontal_flip=True,
                vertical_flip=True,
                hsv_aug_sigma=(0.05, 0.05, 0.05),
                rot90=3
                )
        else:
            datagen = PatchDataGenerator()
    if mode is "slide":
        if augment:
            datagen = SlideDataGenerator(
                featurewise_center=False,
                featurewise_std_normalization=False,
                zca_whitening=False,
                horizontal_flip=True,
                vertical_flip=True,
                hsv_aug_sigma=(0.05, 0.04, 0.04),
                rot90=3
                )
        else:
            datagen = SlideDataGenerator()
    return datagen
