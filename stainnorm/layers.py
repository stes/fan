""" Extensions to lasagne, in terms of convenience functions and new layer types
"""

__author__ = "sschneider"
__email__ = "steffen.schneider@rwth-aachen.de"

from collections import OrderedDict
import pickle
import numpy as np
import theano
from theano import tensor as T
import pickle

import lasagne as nn
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer, InputLayer,\
                           DenseLayer, NonlinearityLayer, BiasLayer, \
                           BatchNormLayer, Upscale2DLayer, ExpressionLayer, \
                           TransposedConv2DLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.nonlinearities import rectify, linear
from lasagne.nonlinearities import sigmoid, tanh
import h5py


###--------------------------------------------------------------------------------
# Implementation of SSIM measure in Theano

from scipy.ndimage.filters import gaussian_filter, uniform_filter
from theano.tensor.signal import conv

def get_gaussian(nsig=1.5, kernlen=13):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return theano.shared(kernel.astype("float32"), borrow=True)

def get_uniform(kernlen=13):
    """Returns a 2D Uniform kernel array."""
    kernel_raw = np.ones((kernlen, kernlen))
    kernel = kernel_raw/kernel_raw.sum()
    return theano.shared(kernel.astype("float32"), borrow=True)

def conv2d(x, kernel, conv=conv.conv2d, *args, **kwargs):
    b,c,d0,d1 = x.shape
    y = conv(x.reshape((b*c,d0,d1)), kernel, *args, **kwargs)
    d0, d1 = y.shape[1:]
    return y.reshape( (b,c,d0,d1) )


def compare_ssim(X, Y, win_size=None, gradient=False,
                 data_range=None, multichannel=False, gaussian_weights=False,
                 full=False, dynamic_range=None, **kwargs):


    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)

    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")

    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        if gaussian_weights:
            win_size = 11  # 11 to match Wang et. al. 2004
        else:
            win_size = 7   # backwards compatibility

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        data_range = 1

    ndim = X.ndim


    if gaussian_weights:
        filters = get_gaussian(sigma, 13)
    else:
        filters = get_uniform(win_size)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004


    # compute (weighted) means
    ux = conv2d(X, filters)
    uy = conv2d(Y, filters)

    # compute (weighted) variances and covariances
    uxx = conv2d(X * X, filters)
    uyy = conv2d(Y * Y, filters)
    uxy = conv2d(X * Y, filters)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = T.sqr(K1 * R)
    C2 = T.sqr(K2 * R)

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    return S


###--------------------------------------------------------------------------------
# Lasagne layer extensions

class ElemwiseProdLayer(ElemwiseMergeLayer):
    """
    This layer performs an elementwise product of its input layers.
    It requires all input layers to have the same output shape.
    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes,
        with all incoming shapes being equal
    coeffs: list or scalar
        A same-sized list of coefficients, or a single coefficient that
        is to be applied to all instances. By default, these will not
        be included in the learnable parameters of this layer.
    cropping : None or [crop]
        Cropping for each input axis. Cropping is described in the docstring
        for :func:`autocrop`
    Notes
    -----
    Depending on your architecture, this can be used to avoid the more
    costly :class:`ConcatLayer`. For example, instead of concatenating layers
    before a :class:`DenseLayer`, insert separate :class:`DenseLayer` instances
    of the same number of output units and add them up afterwards. (This avoids
    the copy operations in concatenation, but splits up the dot product.)
    """

    prod = lambda x, y: x * y

    def __init__(self, incomings, coeffs=1, cropping=None, **kwargs):
        super(ElemwiseProdLayer, self).__init__(incomings, ElemwiseProdLayer.prod,
                                                cropping=cropping, **kwargs)
        if isinstance(coeffs, list):
            if len(coeffs) != len(incomings):
                raise ValueError("Mismatch: got %d coeffs for %d incomings" %
                                 (len(coeffs), len(incomings)))
        else:
            coeffs = [coeffs] * len(incomings)

        self.coeffs = coeffs

    def get_output_for(self, inputs, **kwargs):
        # if needed multiply each input by its coefficient
        inputs = [input * coeff if coeff != 1 else input
                  for coeff, input in zip(self.coeffs, inputs)]

        # pass scaled inputs to the super class for multiplication
        return super(ElemwiseProdLayer, self).get_output_for(inputs, **kwargs)


###--------------------------------------------------------------------------------
# Convenience functions

def transpose(incoming, conv, nonlinearity, *args, **kwargs):
    """ Convenience function to transpose a convolutional layer
    and use weight tying
    """
    return TransposedConv2DLayer(incoming, conv.input_shape[1],
                                 conv.filter_size, stride=conv.stride,
                                 crop=conv.pad, W=conv.W,
                                 flip_filters=not conv.flip_filters,
                                 nonlinearity=nonlinearity, *args,
                                 **kwargs)

def upsample_bilinear(layer, nb_kernels, ratio=2):
    """
    """

    def build_bilinear_kernel(ratio):
        half_kern = np.arange(1, ratio + 1)
        kern = np.concatenate([half_kern, half_kern[-2::-1]])

        kern = kern / ratio
        kern = kern[:,np.newaxis] * kern[np.newaxis, :]
        return kern

    kernel = build_bilinear_kernel(ratio=ratio)
    kernel_shape = (nb_kernels, nb_kernels) + kernel.shape
    W_init = np.zeros(kernel_shape)
    W_init[range(nb_kernels), range(nb_kernels), :, :] = kernel
    W_init = theano.shared(np.float32(W_init))

    return nn.layers.TransposedConv2DLayer(layer, nb_kernels, ratio*2+1, stride=(ratio, ratio), W=W_init, b=None, nonlinearity=None)


###--------------------------------------------------------------------------------
# LSTM unit

def upsample(layer, scale, mode="repeat"):
    """ Upsampling by repetition or bilinear upsampling
    """
    if mode in ["repeat", "dilate"]:
        return Upscale2DLayer(layer, scale, mode=mode)
    elif mode in ["bilinear"]:
        nb_kernels = nn.layers.get_output_shape(layer)[1]
        return upsample_bilinear(layer, nb_kernels, ratio=scale)
    raise ValueError("Invalid mode: " + str(mode))


def fan_module_simple(inp, net, prefix, features, nb_filter, scale, upsampling_strategy="repeat"):
    r""" Implementation for simple LSTM block for feature based manipulation

    Takes input x and features and performs pixelwise manipulation of inp:
    $$
    y = x \sigma(f(z)) + \tanh(g(z)) (1 - \sigma(f(z)))
    $$

    $f$ and $g$ are functions implemented by 1x1 convolutions followed by upsampling
    to match the dimension of $x$.

    """

    # Input gate directly derived from feature representation. Sigmoid rescales to 0...1
    input_gate = ConvLayer(features, nb_filter, 1, pad=0, flip_filters=False,
                           nonlinearity=sigmoid, b=nn.init.Constant(0.5))

    # Addition gate uses inverse activation from input gate
    addition_gate = ExpressionLayer(input_gate, lambda x: 1 - x)
    addition_info = ConvLayer(features, nb_filter, 1, pad=0, flip_filters=False,
                              nonlinearity=tanh)
    addition = ElemwiseProdLayer([addition_gate, addition_info])

    input_gate_upsampled = upsample(input_gate, scale, mode=upsampling_strategy)
    addition_gate_upsampled = upsample(addition, scale, mode=upsampling_strategy)

    x_forget = ElemwiseProdLayer([inp, input_gate_upsampled],
                                 cropping=(None, None, "center", "center"))
    x_added = ElemwiseSumLayer([x_forget, addition_gate_upsampled],
                               cropping=(None, None, "center", "center"))

    ll = [input_gate, addition_gate, addition_info, addition, input_gate_upsampled,
          addition_gate_upsampled, x_forget, x_added]
    layers = locals()
    net.update({prefix + "/" + k: layers[k] for k in layers.keys() if layers[k] in ll})

    return x_added

def fan_module_improved(inp, net, prefix, features, nb_filter, scale, upsampling_strategy="repeat"):
    r""" Implementation for simple LSTM block for feature based manipulation

    Takes input x and features and performs pixelwise manipulation of inp:
    $$
    y = x \sigma(f(z)) + \tanh(g(z)) (1 - \sigma(f(z)))
    $$

    $f$ and $g$ are functions implemented by 1x1 convolutions followed by upsampling
    to match the dimension of $x$.

    """

    # Input gate directly derived from feature representation. Sigmoid rescales to 0...1
    input_gate = ConvLayer(features, nb_filter, 1, pad=0, flip_filters=False,
                           nonlinearity=sigmoid, b=nn.init.Constant(0.5))

    # Addition gate uses inverse activation from input gate
    addition = ConvLayer(features, nb_filter, 1, pad=0, flip_filters=False,
                                  nonlinearity=rectify)

    input_gate_upsampled = upsample(input_gate, scale, mode=upsampling_strategy)
    addition_gate_upsampled = upsample(addition, scale, mode=upsampling_strategy)

    x_forget = ElemwiseProdLayer([inp, input_gate_upsampled],
                                 cropping=(None, None, "center", "center"))
    x_added = ElemwiseSumLayer([x_forget, addition_gate_upsampled],
                               cropping=(None, None, "center", "center"))

    ll = [input_gate, addition, input_gate_upsampled,
          addition_gate_upsampled, x_forget, x_added]
    layers = locals()
    net.update({prefix + "/" + k: layers[k] for k in layers.keys() if layers[k] in ll})

    return x_added


###--------------------------------------------------------------------------------
# Normalizer Networks

def normalize(x):
    """ Normalize the given tensor for use with the VGG networks
    """
    MEAN_VALUES = np.array([104, 117, 123])
    means = theano.shared(MEAN_VALUES.astype("float32"))
    return x[:, ::-1, :, :] - means[np.newaxis, :, np.newaxis, np.newaxis]


def get_features(inp_layer, pad=0):
    """ Return VGG19 feature extractor with ImageNet weights.

    The VGG19 feature extractor consists of five blocks.
    The first two blocks consist of two layers each, while the latter three are
    made of four stacked layers.

    Stacked 3x3 convolution followed by convolution used everywhere.

    Notes
    -----
    Code taken from https://github.com/Lasagne/Recipes/blob/master/modelzoo/vgg19.py
    Original source: https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
    License: see http://www.robots.ox.ac.uk/~vgg/research/very_deep/

    VGG19 weights : [[https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19.pkl]]
    md5sum        : ``cb8ee699c50a64f8fef2a82bfbb307c5``
    """
    # Note: tweaked to use average pooling instead of maxpooling
    net = OrderedDict()
    net['conv1_1'] = ConvLayer(inp_layer, 64, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['pool1'] = PoolLayer(net['conv1_2'], 2, mode='average_exc_pad')
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['pool2'] = PoolLayer(net['conv2_2'], 2, mode='average_exc_pad')
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['pool3'] = PoolLayer(net['conv3_4'], 2, mode='average_exc_pad')
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    #net['pool4'] = PoolLayer(net['conv4_4'], 2, mode='average_exc_pad')
    #net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    #net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    #net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    #net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=pad, flip_filters=False, nonlinearity=rectify)
    #net['pool5'] = PoolLayer(net['conv5_4'], 2, mode='average_exc_pad')

    nb_params = len(nn.layers.get_all_params(net['conv4_4']))

    values = pickle.load(open('vgg19_normalized.pkl', 'rb'), encoding='latin1')['param values']
    nn.layers.set_all_param_values(net['conv4_4'], values[:nb_params])

    return net
