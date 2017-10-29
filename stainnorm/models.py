""" Model collection from first batch of experiments

Notes
-----

Note that all models collected here are designed for fixed 256x256 patch normalization.
Predifined shape helps theano to build more computationally efficient graph, so highly
recommended to use this during training time.
For testing, it is of course better to have undefined spatial dimensions, however this is
(right now) not the primary goal of this code collections and hence not implemented.

Contributing
------------

For later comparison of approaches, please continue with numbering and *do not* rename
existing functions, as this will confuse loading of stored weights. Whereever datasets
and/or weight files are used, a suitable hash has to be provided.

Ressources
----------

- VGG16 weights : [["https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl"]]
  License       : Non-commercial use only
  md5sum        : 57858ec9bb7435e99c78a0520e6c5d3e

- VGG19 weights : [["https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg19_normalized.pkl"]]
  License       : Non-commercial use only
  md5sum        : cb8ee699c50a64f8fef2a82bfbb307c5

Changelog
---------

v0.3 (..-01-2017)
- Added bilinear upsampling, especially used in the feature path

v0.2 (09-12-2016)
- Added the Feature-Aware Normalization (FAN) layer, replacing the final batch norm layer
  in Baseline 1
- The model is up and running, however learning the FAN parameters is rather slow and not
  yet evaluated, but it seems to work :-)

v0.1 (05-12-2016)
- Added four baseline models with varying amount of model complexity
- Baselines 1 and 2 confirm that the batch norm layer on the output alone has a huge
  impact on system performance
- Baseline 3 is the old approach using the first VGG block
"""

__author__ = "sschneider"
__email__ = "steffen.schneider@rwth-aachen.de"

from collections import OrderedDict
import pickle

import lasagne as nn
from lasagne.layers import InputLayer, NonlinearityLayer, BatchNormLayer
try:
    from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
except:
    print("Failed to use GPU implementations of Conv and Pool Layers")
    from lasagne.layers import Pool2DLayer as PoolLayer
    from lasagne.layers import Conv2DLayer as ConvLayer

from lasagne.layers import Upscale2DLayer
from lasagne.layers import ExpressionLayer, TransposedConv2DLayer
from lasagne.nonlinearities import rectify, linear
from lasagne.layers import get_output_shape

from . import layers, tools
from .layers import fan_module_simple, fan_module_improved
from .layers import get_features, normalize, transpose
from .featurenorm import FeatureAwareNormLayer

__all__ = ['build_baseline1_small',
           'build_baseline2_feats',
           'build_baseline3_vgg',
           'build_baseline4_fan',
           'build_baseline5_fan',
           'build_baseline6_fan_fan',
           'build_resnet7_fan',
           'build_baseline8_fan_bilinear',
           'build_baseline9_fan_fan_bilinear',
           'build_finetuned1_fan',
           'build_finetuned2_fan',
           'build_big_fan',
           'build_fan_reworked']


###
# Small Baseline Model
def build_baseline1_small(input_var):
    """ Most simplistic model possible. Effectively only uses last batch norm layer
    """
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer((None, 3, tools.INP_PSIZE, tools.INP_PSIZE), input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))
    last = net["middle"] = ConvLayer(last, 3, 1, nonlinearity=linear)
    last = net["bn"] = BatchNormLayer(last, beta=nn.init.Constant(128.), gamma=nn.init.Constant(25.))

    return last, net


def build_baseline2_feats(input_var, nb_filter=96):
    """ Slightly more complex model. Transform x to a feature space first
    """
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer((None, 3, tools.INP_PSIZE, tools.INP_PSIZE), input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"] = BatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"] = BatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # Decoder as before
    last = net["deconv1_2"] = TransposedConv2DLayer(last, net["conv1_2"].input_shape[1],
                                                    net["conv1_2"].filter_size, stride=net["conv1_2"].stride,
                                                    crop=net["conv1_2"].pad,
                                                    W=net["conv1_2"].W, flip_filters=not net["conv1_2"].flip_filters,
                                                    nonlinearity=None)
    last = net["deconv1_1"] = TransposedConv2DLayer(last, net["conv1_1"].input_shape[1],
                                                    net["conv1_1"].filter_size, stride=net["conv1_1"].stride,
                                                    crop=net["conv1_1"].pad,
                                                    W=net["conv1_1"].W, flip_filters=not net["conv1_1"].flip_filters,
                                                    nonlinearity=None)

    last = net["bn"] = BatchNormLayer(last, beta=nn.init.Constant(128.), gamma=nn.init.Constant(25.))

    return last, net


###
# VGG + LSTM Model
def build_baseline3_vgg(input_var, nb_filter=64):
    net = OrderedDict()

    def get_weights(file):
        with open(file, "rb") as f:
            vgg16 = pickle.load(f, encoding="latin-1")
            weights = vgg16['param values']
        return weights[0], weights[1], weights[2], weights[3]

    # Input, standardization
    last = net['input'] = InputLayer((None, 3, tools.INP_PSIZE, tools.INP_PSIZE), input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    net['features_s8'] = get_features(last)["conv4_1"]
    net['features_s4'] = get_features(last)["conv3_3"]

    # Pretrained Encoder as before
    W1, b1, W2, b2 = get_weights("vgg16.pkl")
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 3, pad=1, flip_filters=False,
                                      nonlinearity=linear, W=W1, b=b1)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 3, pad=1, flip_filters=False,
                                      nonlinearity=linear, W=W2, b=b2)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["pool"] = PoolLayer(last, 2, mode="average_exc_pad")

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = fan_module_simple(last, net, "s8", net['features_s8'], nb_filter=64, scale=4)
    last = fan_module_simple(last, net, "s4", net['features_s4'], nb_filter=64, scale=2)

    # Decoder as before
    last = net["unpool"] = Upscale2DLayer(last, 2)
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    last = net["bn"] = BatchNormLayer(last, beta=nn.init.Constant(128.), gamma=nn.init.Constant(25.))

    return last, net


###
# FULL LSTM Model
def build_baseline4_fan(input_var, nb_filter=96, input_size=(None,3,tools.INP_PSIZE,tools.INP_PSIZE)):
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer(input_size, input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    net['features_s8'] = get_features(last)["conv4_1"]
    net['features_s4'] = get_features(last)["conv3_3"]

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"] = BatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"] = BatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = net["bn1"] = BatchNormLayer(last)
    last = fan_module_simple(last, net, "s8", net['features_s8'],
                              nb_filter=nb_filter, scale=8)
    last = net["bn2"] = BatchNormLayer(last)
    last = fan_module_simple(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4)

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    last = net["bn"] = BatchNormLayer(last, beta=nn.init.Constant(128.), gamma=nn.init.Constant(25.))

    return last, net


###
# Model with new Feature Norm Layer
def build_baseline5_fan(input_var):
    # TODO remove these imports + move relevant parts to layers.py once everything is
    # up and running
    import theano.tensor as T
    import numpy as np
    """ Using Baseline 1 with the novel FAN layer.

    VGG conv4_1 is used for feature extraction
    """
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer((None, 3, tools.INP_PSIZE, tools.INP_PSIZE), input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    net['features_s8'] = get_features(last)["conv4_1"]
    net['features'] = Upscale2DLayer(net["features_s8"], 8)
    net['mask'] = ExpressionLayer(net["features"], lambda x: 1. * T.eq(x, x.max(axis=1, keepdims=True)))

    last = net["middle"] = ConvLayer(last, 3, 1, nonlinearity=linear)
    last = net["fan"] = FeatureAwareNormLayer((last, net['mask']), beta=nn.init.Constant(np.float32(128.)),
                                              gamma=nn.init.Constant(np.float32(25.)))

    return last, net


def build_baseline6_fan_fan(input_var, nb_filter=96):
    net = OrderedDict()

    import theano.tensor as T
    import numpy as np

    # Input, standardization
    last = net['input'] = InputLayer((None, 3, tools.INP_PSIZE, tools.INP_PSIZE), input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    net['features_s8'] = get_features(last)["conv4_1"]
    net['features_s4'] = get_features(last)["conv3_3"]
    net['mask'] = ExpressionLayer(Upscale2DLayer(net["features_s8"], 8),
                                  lambda x: 1. * T.eq(x, x.max(axis=1, keepdims=True)))

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"] = BatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"] = BatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = net["fan1"] = FeatureAwareNormLayer((last, net['mask']))
    last = fan_module_simple(last, net, "s8", net['features_s8'],
                              nb_filter=nb_filter, scale=8)
    last = net["fan2"] = FeatureAwareNormLayer((last, net['mask']))
    last = fan_module_simple(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4)

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    last = net["fan"] = FeatureAwareNormLayer((last, net['mask']), beta=nn.init.Constant(np.float32(128.)),
                                              gamma=nn.init.Constant(np.float32(25.)))

    return last, net


def build_resnet7_fan(input_var, nb_filter=96):
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer((None, 3, tools.INP_PSIZE, tools.INP_PSIZE), input_var=input_var)

    # Note: normalization should not be necessary for the ResNet model!
    # last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    res50 = get_resnet50(last)
    net['features_s8'] = res50['res3d_branch2c']
    net['features_s4'] = res50['res2c_branch2c']

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"] = BatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"] = BatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = net["bn1"] = BatchNormLayer(last)
    last = fan_module_simple(last, net, "s8", net['features_s8'],
                              nb_filter=nb_filter, scale=8)
    last = net["bn2"] = BatchNormLayer(last)
    last = fan_module_simple(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4)

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    last = net["bn"] = BatchNormLayer(last, beta=nn.init.Constant(128.), gamma=nn.init.Constant(25.))

    return last, net


###
# FULL LSTM Model with Bilinar upsampling
def build_baseline8_fan_bilinear(input_var, nb_filter=96):
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer((None, 3, tools.INP_PSIZE, tools.INP_PSIZE), input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    net['features_s8'] = get_features(last)["conv4_1"]
    net['features_s4'] = get_features(last)["conv3_3"]

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"] = BatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"] = BatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = net["bn1"] = BatchNormLayer(last)
    last = fan_module_simple(last, net, "s8", net['features_s8'],
                              nb_filter=nb_filter, scale=8, upsampling_strategy="bilinear")
    last = net["bn2"] = BatchNormLayer(last)
    last = fan_module_simple(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4, upsampling_strategy="bilinear")

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    last = net["bn"] = BatchNormLayer(last, beta=nn.init.Constant(128.), gamma=nn.init.Constant(25.))

    return last, net


def build_baseline9_fan_fan_bilinear(input_var, nb_filter=96):
    net = OrderedDict()

    import theano.tensor as T
    import numpy as np

    # Input, standardization
    last = net['input'] = InputLayer((None, 3, tools.INP_PSIZE, tools.INP_PSIZE), input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    net['features_s8'] = get_features(last)["conv4_1"]
    net['features_s4'] = get_features(last)["conv3_3"]
    net['mask'] = ExpressionLayer(layers.upsample(net["features_s8"], 8, mode="bilinear"),
                                  lambda x: 1. * T.eq(x, x.max(axis=1, keepdims=True)))

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"] = BatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"] = BatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = net["fan1"] = FeatureAwareNormLayer((last, net['mask']))
    last = fan_module_simple(last, net, "s8", net['features_s8'],
                              nb_filter=nb_filter, scale=8, upsampling_strategy="bilinear")
    last = net["fan2"] = FeatureAwareNormLayer((last, net['mask']))
    last = fan_module_simple(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4, upsampling_strategy="bilinear")

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    last = net["fan"] = FeatureAwareNormLayer((last, net['mask']), beta=nn.init.Constant(np.float32(128.)),
                                              gamma=nn.init.Constant(np.float32(25.)))

    return last, net

###
# FULL LSTM Model
def build_finetuned1_fan(input_var, nb_filter=96, input_size=(None,3,tools.INP_PSIZE,tools.INP_PSIZE)):
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer(input_size, input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    # TODO this is clearly a bug. only for compatibility reasons. remove once all weights are converted
    net['features_s8'] = get_features(last)["conv4_1"]
    net['features_s4'] = get_features(last)["conv3_3"]

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"]   = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"]   = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = net["bn1"] = layers.NonUpdateBatchNormLayer(last)
    last = fan1= fan_module_simple(last, net, "s8", net['features_s8'],
                              nb_filter=nb_filter, scale=8)
    last = net["bn2"] = layers.NonUpdateBatchNormLayer(last)
    last = fan2= fan_module_simple(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4)

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    last = net["bn"] = layers.FixedBatchNormLayer(last)

    ## for debugging: decoders after each LSTM module
    def debug_connection(last):
        last = transpose(last, net["conv1_2"], nonlinearity=None, b=net['deconv1_2'].b)
        last = transpose(last, net["conv1_1"], nonlinearity=None, b=net['deconv1_1'].b)
        last = layers.FixedBatchNormLayer(last, beta=net['bn'].beta, gamma=net['bn'].gamma, mean=net['bn'].mean, inv_std=net['bn'].inv_std)

        return last

    debug1 = debug_connection(fan1)
    debug2 = debug_connection(fan2)

    weights = "170123_runs/run_H.E.T._1485012575.4045253/3.npz"
    data = tools.load_weights(last, weights)

    return last, net, debug1, debug2


###
# FULL LSTM Model
def build_finetuned2_fan(input_var, nb_filter=96, input_size=(None,3,tools.INP_PSIZE,tools.INP_PSIZE)):
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer(input_size, input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    # TODO this is clearly a bug. only for compatibility reasons. remove once all weights are converted
    net['features_s8'] = get_features(last)["conv4_1"]
    net['features_s4'] = get_features(last)["conv3_3"]

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"]   = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"]   = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = net["bn1"] = layers.NonUpdateBatchNormLayer(last)
    last = fan_module_simple(last, net, "s8", net['features_s8'],
                              nb_filter=nb_filter, scale=8)
    last = net["bn2"] = layers.NonUpdateBatchNormLayer(last)
    last = fan_module_simple(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4)

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    last = net["bn"] = layers.FixedBatchNormLayer(last)

    weights = "170123_runs/run_H.E.T._1485012575.4045253/3.npz"
    data = tools.load_weights(last, weights)

    return last, net


###
# FULL LSTM Model
def build_big_fan(input_var, nb_filter=96, input_size=(None,3,tools.INP_PSIZE,tools.INP_PSIZE)):
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer(input_size, input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    f = get_features(last)
    net['features_s8'] = f["conv4_1"]
    net['features_s4'] = f["conv3_3"]

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"]   = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"]   = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = net["bn1"] = layers.NonUpdateBatchNormLayer(last)
    last = fan_module_simple(last, net, "s8", net['features_s8'],
                              nb_filter=nb_filter, scale=8)
    last = net["bn1"] = layers.NonUpdateBatchNormLayer(last)
    last = fan_module_simple(last, net, "s8", net['features_s8'],
                              nb_filter=nb_filter, scale=8)
    last = net["bn3"] = layers.NonUpdateBatchNormLayer(last)
    last = fan_module_simple(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4)
    last = net["bn4"] = layers.NonUpdateBatchNormLayer(last)
    last = fan_module_simple(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4)
    last = net["bn5"] = layers.NonUpdateBatchNormLayer(last)

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    return last, net

###
# FULL Feature Aware Normalization Model
def build_fan_reworked(input_var, nb_filter=16, input_size=(None,3,tools.INP_PSIZE,tools.INP_PSIZE)):
    net = OrderedDict()

    # Input, standardization
    last = net['input'] = InputLayer(input_size, input_var=input_var)
    last = net['norm'] = ExpressionLayer(last, lambda x: normalize(x))

    # load feature encoder
    feats = get_features(last)
    net['features_s8_1'] = feats["conv4_4"]
    net['features_s8_2'] = feats["conv4_1"]
    net['features_s4'] = feats["conv3_3"]

    # Pretrained Encoder as before
    last = net["conv1_1"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_1"]   = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"]   = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # feature aggregation at multiple scales
    last = net["bn1"] = layers.NonUpdateBatchNormLayer(last, beta=None, gamma=None)
    last = fan_module_improved(last, net, "s8_1", net['features_s8_1'],
                              nb_filter=nb_filter, scale=8, upsampling_strategy="repeat")
    last = net["bn2"] = layers.NonUpdateBatchNormLayer(last, beta=None, gamma=None)
    last = fan_module_improved(last, net, "s8_2", net['features_s8_2'],
                              nb_filter=nb_filter, scale=8, upsampling_strategy="repeat")
    last = net["bn3"] = layers.NonUpdateBatchNormLayer(last, beta=None, gamma=None)
    last = fan_module_improved(last, net, "s4", net['features_s4'],
                              nb_filter=nb_filter, scale=4, upsampling_strategy="repeat")
    # unclear if Fixed, NonUpdate or Regular Layer will work best...
    last = net["bn4"] = BatchNormLayer(last)

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    return last, net
