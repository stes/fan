from collections import OrderedDict

import lasagne as nn
from lasagne.layers import InputLayer, NonlinearityLayer, BatchNormLayer
from lasagne.layers.dnn import Pool2DDNNLayer as PoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Upscale2DLayer
from lasagne.layers import ExpressionLayer, TransposedConv2DLayer
from lasagne.nonlinearities import rectify, linear
from lasagne.layers import get_output_shape

import layers
from layers import lstm_module_simple, get_features, normalize, transpose, get_resnet50, lstm_module_improved
from featurenorm import FeatureAwareNormLayer
import pickle
import tools


def build_finetuned1_lstm(input_var, nb_filter=96, input_size=(None, 3, tools.INP_PSIZE, tools.INP_PSIZE)):
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
    last = net["bn1_1"] = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"] = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # Modified Middle Part
    last = net["middle"] = ConvLayer(last, nb_filter, 1, nonlinearity=linear)

    # feature aggregation at multiple scales
    last = net["bn1"] = layers.NonUpdateBatchNormLayer(last)
    last = lstm1 = lstm_module_simple(last, net, "s8", net['features_s8'],
                                      nb_filter=nb_filter, scale=8)
    last = net["bn2"] = layers.NonUpdateBatchNormLayer(last)
    last = lstm2 = lstm_module_simple(last, net, "s4", net['features_s4'],
                                      nb_filter=nb_filter, scale=4)

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)
    last = net["bn"] = layers.FixedBatchNormLayer(last)

    ## for debugging: decoders after each LSTM module
    def debug_connection(l):
        l = transpose(l, net["conv1_2"], nonlinearity=None, b=net['deconv1_2'].b)
        l = transpose(l, net["conv1_1"], nonlinearity=None, b=net['deconv1_1'].b)
        l = layers.FixedBatchNormLayer(l, beta=net['bn'].beta, gamma=net['bn'].gamma, mean=net['bn'].mean,
                                       inv_std=net['bn'].inv_std)

        return l

    debug = [debug_connection(l) for l in [net["middle"], net["bn1"], lstm1, net["bn2"], lstm2]]

    # features and resulting representations
    debug.append(net["s8/addition"])
    debug.append(net["s8/input_gate"])

    debug.append(net["s4/addition"])
    debug.append(net["s4/input_gate"])

    return last, net, debug


def build_lstm_reworked(input_var, nb_filter=16, input_size=(None, 3, tools.INP_PSIZE, tools.INP_PSIZE), debug_connections=True):
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
    last = net["bn1_1"] = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_1"] = NonlinearityLayer(last, nonlinearity=rectify)
    last = net["conv1_2"] = ConvLayer(last, nb_filter, 1, pad=0, flip_filters=False,
                                      nonlinearity=linear)
    last = net["bn1_2"] = layers.NonUpdateBatchNormLayer(last)
    last = net["relu1_2"] = NonlinearityLayer(last, nonlinearity=rectify)

    # feature aggregation at multiple scales
    last = net["bn1"] = layers.NonUpdateBatchNormLayer(last, beta=None, gamma=None)
    last = lstm1 = lstm_module_improved(last, net, "s8_1", net['features_s8_1'],
                                        nb_filter=nb_filter, scale=8)
    last = net["bn2"] = layers.NonUpdateBatchNormLayer(last, beta=None, gamma=None)
    last = lstm2 = lstm_module_improved(last, net, "s8_2", net['features_s8_2'],
                                        nb_filter=nb_filter, scale=8)
    last = net["bn3"] = layers.NonUpdateBatchNormLayer(last, beta=None, gamma=None)
    last = lstm3 = lstm_module_improved(last, net, "s4", net['features_s4'],
                                        nb_filter=nb_filter, scale=4)
    last = net["bn4"] = layers.FixedBatchNormLayer(last)

    # Decoder as before
    last = net["deconv1_2"] = transpose(last, net["conv1_2"], nonlinearity=None)
    last = net["deconv1_1"] = transpose(last, net["conv1_1"], nonlinearity=None)

    def debug_connection(l):
        l = layers.FixedBatchNormLayer(l, beta=net['bn4'].beta, gamma=net['bn4'].gamma,
                                       mean=net['bn4'].mean, inv_std=net['bn4'].inv_std)
        l = transpose(l, net["conv1_2"], nonlinearity=None, b=net['deconv1_2'].b)
        l = transpose(l, net["conv1_1"], nonlinearity=None, b=net['deconv1_1'].b)

        return l

    # tools.load_weights(last, "lstm_improved_H.E.T._1485219840.1135952/12.npz")
    debug = []
    if debug_connections:
        debug = [debug_connection(l) for l in [lstm1, lstm2, lstm3]]
    else:
        debug = [net["relu1_2"], lstm1, lstm2, lstm3, net["bn4"]]

    # features and resulting representations
    debug.append(net["s8_1/addition"])
    debug.append(net["s8_1/input_gate"])

    debug.append(net["s8_2/addition"])
    debug.append(net["s8_2/input_gate"])

    debug.append(net["s4/addition"])
    debug.append(net["s4/input_gate"])

    return last, net, debug
