""" Feature-Aware Normalization
"""

__author__ = "sschneider"
__email__ = "steffen.schneider@rwth-aachen.de"

import theano
from theano import tensor as T
from lasagne.layers.merge import autocrop, MergeLayer
from lasagne import init
import numpy as np
import lasagne as nn


def heavyside(x):
    """ Compute the heavyside function in theano
    
    Notes
    -----
    See https://github.com/Theano/Theano/issues/2459
    """
    return (T.sgn(x) + 1) / 2
    # return T.gt(x, 0)


def feature_statistics(x, m, eps=1e-4):
    """ Given an input tensor and a binary mask, compute partition based statistics

    Efficiently compute statistics over partitions of the input image using theano's
    tensordot Op.

    Parameters
    ----------
    x : tensor4, shape ``B,C,I,J``
    m : tensor4, shape ``B,K,I,J``

    Returns
    -------
        list of mean and inverted standard deviation. shape for each ``[C, K]``
    """
    m_sum = m.sum(axis=(0, 2, 3)) + eps

    mean = T.tensordot(x, m, axes=((0, 2, 3), (0, 2, 3))) / m_sum

    # Variance calculation 1, faster but numerically instable:
    # var = T.tensordot(T.sqr(x),m,axes=((0,2,3),(0,2,3))) / m_sum - T.sqr(mean)

    # Variant 2 (computation of mean tensordot will probably be optimized by theano,
    # so use this solution for now.
    means = T.tensordot(mean, m, axes=((1,), (1,))).transpose((1, 0, 2, 3))
    var = T.tensordot(T.sqr(x - means), m, axes=((0, 2, 3), (0, 2, 3))) / m_sum

    inv_std = T.inv(T.sqrt(var + eps))

    return mean, inv_std


def feature_normalization(x, m, mean, inv_std, gamma, beta):
    """ Feature-Aware Normalization
    
    Given an input and a mask matching over all but the second dimension, use the
    provided image statistics to normalize image pixels in each region.

    Stats can be computed by ``feature_statistics``.

    Parameters
    ----------
    x       : tensor4, shape ``B,C,I,J``, input feature maps/images
    m       : tensor4, shape ``B,K,I,J``, binary mask with non-overlapping channels
    mean    : matrix, shape ``C,K``, input means computed over input partitions
    inv_std : matrix, shape ``C,K``, inverse standard deviation computed over input partitions
    gamma   : matrix, shape ``C,K``, scale parameter
    beta    : matrix, shape ``C,K``, bias parameter

    Returns
    -------
        tensor4, shape ``B,C,I,J``, normalized image
    """
    # [C,K].(1,1)*[B,K,I,J] = [C,B,I,J]
    means = T.tensordot(mean, m, axes=((1,), (1,))).transpose((1, 0, 2, 3))
    mult = T.tensordot((gamma), m, axes=((1,), (1,))).transpose((1, 0, 2, 3)) \
           * T.tensordot((inv_std), m, axes=((1,), (1,))).transpose((1, 0, 2, 3))
    bias = T.tensordot(beta, m, axes=((1,), (1,))).transpose((1, 0, 2, 3))
    normalized = (x - means) * mult + bias
    return normalized


class FeatureAwareNormLayer(MergeLayer):
    """ Lasagne module for the Feature-Aware Normalization (FAN) Layer 

    For now, refer to docs of ``lasagne.layers.BatchNormLayer``.
    This layer basically computes the same metrics, but accepts a mask of same
    dimensions as the image (except for the feature dimension, i.e., dimension 1).

    BN stats are computed over all partitions of the image and finally combined.
    """

    def __init__(self, incomings, epsilon=1e-4, alpha=0.1,
                 beta=init.Constant(0), gamma=init.Constant(1),
                 mean=init.Constant(0), inv_std=init.Constant(1), **kwargs):

        super(FeatureAwareNormLayer, self).__init__(incomings, **kwargs)

        self.input_axes = (0,) + tuple(range(2, len(self.input_shapes[0])))
        self.feature_axes = (0,) + tuple(range(2, len(self.input_shapes[1])))

        self.axes = (0, 2, 3)

        self.epsilon = epsilon
        self.alpha = alpha

        # create parameters, ignoring all dimensions in axes
        shape = [self.input_shapes[0][1], self.input_shapes[1][1]]
        if any(size is None for size in shape):
            raise ValueError("BatchNormLayer needs specified input sizes for "
                             "all axes not normalized over.")
        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, shape, 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, shape, 'gamma',
                                        trainable=True, regularizable=True)

        self.mean = self.add_param(mean, shape, 'mean',
                                   trainable=False, regularizable=False)
        self.inv_std = self.add_param(inv_std, shape, 'inv_std',
                                      trainable=False, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        B1, C, I1, J1 = input_shapes[0]
        B2, K, I2, J2 = input_shapes[1]

        output_shape = [B1, C, min(I1, I2), min(J1, J2)]
        return output_shape

    def get_output_for(self, inputs, deterministic=False,
                       batch_norm_use_averages=None,
                       batch_norm_update_averages=None, **kwargs):

        input, features = autocrop(inputs, cropping=(None, None, "center", "center"))

        input_mean, input_inv_std = feature_statistics(input, features)

        # Decide whether to use the stored averages or mini-batch statistics
        if batch_norm_use_averages is None:
            batch_norm_use_averages = deterministic
        use_averages = batch_norm_use_averages

        if use_averages:
            mean = self.mean
            inv_std = self.inv_std
        else:
            mean = input_mean
            inv_std = input_inv_std

        # Decide whether to update the stored averages
        if batch_norm_update_averages is None:
            batch_norm_update_averages = not deterministic
        update_averages = batch_norm_update_averages

        if update_averages:
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(input.ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta
        gamma = 1 if self.gamma is None else self.gamma
        mean = mean
        inv_std = inv_std

        # normalize x, m, mean, inv_std, gamma,beta
        normalized = feature_normalization(input, features, mean, inv_std, gamma, beta)
        return normalized


if __name__ == '__main__':
    # run some tests
    # TODO move to test suite once this layer works

    def create_fake_partitions(nb_samples, nb_features, nb_channels, nb_pixels):
        partitions = []
        mask = np.random.normal(0, 1, size=(nb_samples, nb_features, nb_pixels, nb_pixels))
        mask = (mask == mask.max(axis=(1,), keepdims=True))
        print(mask.shape)
        assert np.allclose(mask.max(axis=1), 1), mask.max(axis=1)
        assert np.allclose(mask.sum(axis=1), 1), mask.sum(axis=1)
        assert (mask.sum(axis=(0, 2, 3)) > 0).all()
        assert (mask.sum() == np.array(mask.shape).prod() / nb_features).all(), mask.sum()

        x_vals = np.random.normal(0, 10, size=(nb_samples, nb_channels, nb_pixels, nb_pixels)).astype("float32")

        return x_vals, mask


    nb_channels = 3
    nb_features = 20
    nb_pixels = 128
    nb_samples = 10
    nb_pixels_features = 21

    # setup theano function for testing
    x = T.tensor4()
    f = T.tensor4()

    inp = nn.layers.InputLayer([nb_samples, nb_channels, nb_pixels, nb_pixels], x)
    feats = nn.layers.InputLayer([nb_samples, nb_features, nb_pixels_features, nb_pixels_features], f)

    norm1 = FeatureAwareNormLayer([inp, feats])
    norm2 = nn.layers.BatchNormLayer(inp)

    outp = nn.layers.get_output([norm1, norm2], deterministic=False)
    fn = theano.function([x, f], outp, allow_input_downcast=True)

    # create inputs and fake features
    x_inp, f_inp = create_fake_partitions(nb_samples, nb_features, nb_channels, nb_pixels)


    # test normalization among partions
    def check():
        x_outp, _ = fn(x_inp, f_inp)
        assert not np.isnan(x_outp).any()
        for mid in range(nb_features):
            b, i, j = np.where(f_inp[:, mid] == 1)
            for c in range(nb_channels):
                std = x_outp[b, c, i, j].std()
                mean = x_outp[b, c, i, j].mean()

                assert np.isclose(std, norm1.gamma.get_value()[c, mid], rtol=1e-3), std
                assert np.isclose(mean, norm1.beta.get_value()[c, mid], atol=1e-6), mean


    check()
    gamma = norm1.gamma.get_value()
    gamma[:, 0] *= 2
    gamma[:, 1] *= 3
    norm1.gamma.set_value(gamma)
    check()
    beta = norm1.beta.get_value()
    beta[:, 0] = 1
    check()

    beta = norm1.beta.get_value()
    gamma = norm1.beta.get_value()
    beta[:, :] = 128
    gamma[:, :] = 25
    check()

    print("All tests passed.")
