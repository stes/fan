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
from lasagne.layers import ElemwiseSumLayer, ElemwiseMergeLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, BiasLayer, BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Upscale2DLayer
from lasagne.layers import ExpressionLayer, TransposedConv2DLayer
from lasagne.nonlinearities import rectify, linear
from lasagne.nonlinearities import sigmoid, tanh
import h5py


###-------------------------------------------------------------------------------- 
# SSIM measure

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
    if mode in ["repeat", "dilate"]:
        return Upscale2DLayer(layer, scale, mode=mode)
    elif mode in ["bilinear"]:
        nb_kernels = nn.layers.get_output_shape(layer)[1]
        return upsample_bilinear(layer, nb_kernels, ratio=scale)
    raise ValueError("Invalid mode: " + str(mode))


def lstm_module_simple(inp, net, prefix, features, nb_filter, scale, upsampling_strategy="repeat"):
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

def lstm_module_improved(inp, net, prefix, features, nb_filter, scale, upsampling_strategy="repeat"):
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
# Normalizer Network(s)

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


### -----------------------------------------------------------------------------
### RESNET Code, borrowed from https://github.com/stes/deepml/blob/master/recipes/resnet.py

class NonUpdateBatchNormLayer(BatchNormLayer):
    """ BN Layer that only uses statistics computed on the current batch
    """

    def __init__(self, incoming, *args, **kwargs):
        super(NonUpdateBatchNormLayer, self).__init__(incoming, *args, **kwargs)
    
    def get_output_for(self, input, **kwargs):
        return super(NonUpdateBatchNormLayer, self).get_output_for(input, batch_norm_use_averages=False, batch_norm_update_averages=False, **kwargs)

class FixedBatchNormLayer(BatchNormLayer):
    """ BN Layer using fixed statistics, with no updates applied to the statistics based on the data
    """

    def __init__(self, incoming, *args, **kwargs):
        super(FixedBatchNormLayer, self).__init__(incoming, *args, **kwargs)
    
    def get_output_for(self, input, deterministic=True, **kwargs):
        return super(FixedBatchNormLayer, self).get_output_for(input, deterministic=True, **kwargs)



def get_resnet50(input_layer):
    """ Build a feature encoder module using the ResNet-x architecture
    Parameters
    ----------
    input_layer : Lasagne layer serving the input to this network module
    net : dict (recommended is collections.OrderedDict) to collect layers
    variant : str, one of "resnet50", "resnet101", "resnet152"
    """
    net = OrderedDict()
    
    def load_weights(fname, net):
        with h5py.File(fname, "r") as ds:
            print('loading weights from', fname)
            for key in net.keys():
                try:
                    # print("Entering group", key)
                    grp = ds[key]
                    for param in net[key].params.keys():
                        p = grp[str(param)]

                        param_shape = param.get_value(borrow=True).shape
                        value_shape = p.shape

                        # print("Adapting:", key, param)
                        param.set_value(p[...])
                        assert (param.get_value() == p[...]).all()
                        # print("\tSuccess.")
                except:
                # except Exception as e:
                    pass
                    # print('ERROR in build_enconder/load_weights', e)

    def build_simple_block(incoming_layer, names,
                           num_filters, filter_size, stride, pad,
                           use_bias=False, nonlin=rectify):
        """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)
        Parameters:
        ----------
        incoming_layer : instance of Lasagne layer
            Parent layer
        names : list of string
            Names of the layers in block
        num_filters : int
            Number of filters in convolution layer
        filter_size : int
            Size of filters in convolution layer
        stride : int
            Stride of convolution layer
        pad : int
            Padding of convolution layer
        use_bias : bool
            Whether to use bias in conlovution layer
        nonlin : function
            Nonlinearity type of Nonlinearity layer
        Returns
        -------
        tuple: (net, last_layer_name)
            net : dict
                Dictionary with stacked layers
            last_layer_name : string
                Last layer name
        """
        net = []
        names = list(names)
        net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

        net.append((
            names[1],
            FixedBatchNormLayer(net[-1][1])
        ))
        if nonlin is not None:
            net.append((
                names[2],
                NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
            ))

        return dict(net), net[-1][0]

    def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                             upscale_factor=4, ix=''):
        """Creates two-branch residual block
        Parameters:
        ----------
        incoming_layer : instance of Lasagne layer
            Parent layer
        ratio_n_filter : float
            Scale factor of filter bank at the input of residual block
        ratio_size : float
            Scale factor of filter size
        has_left_branch : bool
            if True, then left branch contains simple block
        upscale_factor : float
            Scale factor of filter bank at the output of residual block
        ix : int
            Id of residual block
        Returns
        -------
        tuple: (net, last_layer_name)
            net : dict
                Dictionary with stacked layers
            last_layer_name : string
                Last layer name
        """
        simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

        net = OrderedDict()

        # right branch
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
            int(nn.layers.get_output_shape(incoming_layer)[1] * ratio_n_filter), 1, int(1.0 / ratio_size), 0)
        net.update(net_tmp)

        net_tmp, last_layer_name = build_simple_block(
            net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
            nn.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
        net.update(net_tmp)

        net_tmp, last_layer_name = build_simple_block(
            net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
            nn.layers.get_output_shape(net[last_layer_name])[1] * upscale_factor, 1, 1, 0,
            nonlin=None)
        net.update(net_tmp)

        right_tail = net[last_layer_name]
        left_tail = incoming_layer

        # left branch
        if has_left_branch:
            net_tmp, last_layer_name = build_simple_block(
                incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
                int(nn.layers.get_output_shape(incoming_layer)[1] * 4 * ratio_n_filter), 1, int(1.0 / ratio_size), 0,
                nonlin=None)
            net.update(net_tmp)
            left_tail = net[last_layer_name]

        net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
        net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

        return net, 'res%s_relu' % ix

    net['input'] = input_layer

    # BLOCK1 begins here
    sub_net, parent_layer_name = build_simple_block(
        net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],
        64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    parent_layer_name = 'pool1'

    # BLOCK2 begins here
    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4,
                                                              ix='2%s' % c)
        net.update(sub_net)

    # BLOCK3 begins here
    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4,
                                                              ix='3%s' % c)
        net.update(sub_net)

    # BLOCK4 begins here
    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4,
                                                              ix='4%s' % c)
        net.update(sub_net)

    # BLOCK5 begins here
    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(
                net[parent_layer_name], 1.0 / 2, 1.0 / 2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0 / 4, 1, False, 4,
                                                              ix='5%s' % c)
        net.update(sub_net)

    load_weights('resnet50.hdf5', net)

    return net
