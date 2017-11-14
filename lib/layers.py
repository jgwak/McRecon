import numpy as np

# Theano
import collections
import itertools
import theano
import theano.tensor as tensor
from theano.tensor.nnet import conv, conv3d2d, sigmoid
from theano.tensor.signal import pool

from lib.raytrace_voxel import RaytraceVoxelOp
from lib.differentiable_step import DifferentiableStepOp
from lib.utils import max_pool_3d

trainable_params = collections.defaultdict(list)
rs = tensor.shared_randomstreams.RandomStreams(0)


def get_trainable_params():
    global trainable_params
    return trainable_params


def get_rng():
    global rs
    return rs


class Weight(object):

    def __init__(self,
                 w_shape,
                 is_bias,
                 mean=0,
                 std=0.01,
                 filler='msra',
                 fan_in=None,
                 fan_out=None,
                 name=None,
                 param_type=None):
        super(Weight, self).__init__()
        assert (is_bias in [True, False])
        assert param_type is not None
        rng = np.random.RandomState()

        if isinstance(w_shape, collections.Iterable) and not is_bias:
            if len(w_shape) > 1 and len(w_shape) < 5:
                fan_in = np.prod(w_shape[1:])
                fan_out = np.prod(w_shape) / w_shape[1]
                n = (fan_in + fan_out) / 2.
            elif len(w_shape) == 5:
                # 3D Convolution filter
                fan_in = np.prod(w_shape[1:])
                fan_out = np.prod(w_shape) / w_shape[2]
                n = (fan_in + fan_out) / 2.
            else:
                raise NotImplementedError(
                    'Filter shape with ndim > 5 not supported: len(w_shape) = %d' % len(w_shape))
        else:
            n = 1

        if fan_in and fan_out:
            n = (fan_in + fan_out) / 2.

        if filler == 'gaussian':
            self.np_values = np.asarray(rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
        elif filler == 'msra':
            self.np_values = np.asarray(
                rng.normal(mean, np.sqrt(2. / n), w_shape), dtype=theano.config.floatX)
        elif filler == 'xavier':
            scale = np.sqrt(3. / n)
            self.np_values = np.asarray(
                rng.uniform(
                    low=-scale, high=scale, size=w_shape), dtype=theano.config.floatX)
        elif filler == 'constant':
            self.np_values = np.cast[theano.config.floatX](mean * np.ones(
                w_shape, dtype=theano.config.floatX))
        elif filler == 'orth':
            ndim = np.prod(w_shape)
            W = np.random.randn(ndim, ndim)
            u, s, v = np.linalg.svd(W)
            self.np_values = u.astype(theano.config.floatX).reshape(w_shape)
        else:
            raise NotImplementedError('Filler %s not implemented' % filler)

        self.is_bias = is_bias  # Either the weight is bias or not
        self.val = theano.shared(value=self.np_values)
        self.shape = w_shape
        self.name = name

        get_trainable_params()[param_type].append(self)


class InputLayer(object):

    def __init__(self, input_shape, tinput=None):
        self._output_shape = input_shape
        self._input = tinput

    @property
    def output(self):
        if self._input is None:
            raise ValueError('Cannot call output for the layer. Initialize' \
                             + ' the layer with an input argument')
        return self._input

    @property
    def output_shape(self):
        return self._output_shape


class Layer(object):
    ''' Layer abstract class. support basic functionalities.
    If you want to set the output shape, either prev_layer or input_shape must
    be defined.

    If you want to use the computation graph, provide either prev_layer or set_input
    '''

    def __init__(self, prev_layer):
        self._output = None
        self._output_shape = None
        self._prev_layer = prev_layer
        self._input_shape = prev_layer.output_shape
        # Define self._output_shape

    def set_output(self):
        '''Override the function'''
        # set self._output using self._input=self._prev_layer.output
        raise NotImplementedError('Layer virtual class')

    @property
    def output_shape(self):
        if self._output_shape is None:
            raise ValueError('Set output shape first')
        return self._output_shape

    @property
    def output(self):
        if self._output is None:
            self.set_output()
        return self._output


class TensorProductLayer(Layer):

    def __init__(self, prev_layer, n_out, param_type=None, params=None,
                 bias=True):
        super().__init__(prev_layer)
        self._bias = bias
        n_in = self._input_shape[-1]

        if params is None:
            self.W = Weight((n_in, n_out), is_bias=False, param_type=param_type)
            if bias:
                self.b = Weight((n_out,), is_bias=True, param_type=param_type,
                        mean=0.1, filler='constant')
        else:
            self.W = params[0]
            if bias:
                self.b = params[1]

        # parameters of the model
        self.params = [self.W]
        if bias:
            self.params.append(self.b)

        self._output_shape = [self._input_shape[0]]
        self._output_shape.extend(self._input_shape[1:-1])
        self._output_shape.append(n_out)

    def set_output(self):
        self._output = tensor.dot(self._prev_layer.output, self.W.val)
        if self._bias:
            self._output += self.b.val


class BlockDiagonalLayer(Layer):
    """
    Compute block diagonal matrix multiplication efficiently using broadcasting

    Last dimension will be used for matrix multiplication.

    prev_layer.output_shape = N x D_1 x D_2 x ... x D_{n-1} x D_n
    output_shape            = N x D_1 x D_2 x ... x D_{n-1} x n_out
    """

    def __init__(self, prev_layer, n_out, param_type=None, params=None,
                 bias=True):
        super().__init__(prev_layer)
        self._bias = bias
        self._output_shape = list(self._input_shape)
        self._output_shape[-1] = n_out
        self._output_shape = tuple(self._output_shape)

        if params is None:
            self._W_shape = list(self._input_shape[1:])
            self._W_shape.append(n_out)
            self._W_shape = tuple(self._W_shape)
            self.W = Weight(self._W_shape, is_bias=False, param_type=param_type)
            if bias:
                self.b = Weight(self._output_shape[1:], is_bias=True,
                                param_type=param_type, mean=0.1,
                                filler='constant')
        else:
            self.W = params[0]
            if bias:
                self.b = params[1]

        # parameters of the model
        self.params = [self.W]
        if bias:
            self.params.append(self.b)

    def set_output(self):
        self._output = tensor.sum(tensor.shape_padright(self._prev_layer.output) *
                                  tensor.shape_padleft(self.W.val),
                                  axis=-2)
        if self._bias:
            self._output += tensor.shape_padleft(self.b.val)


class AddLayer(Layer):

    def __init__(self, prev_layer, add_layer):
        super().__init__(prev_layer)
        self._output_shape = self._input_shape
        self._add_layer = add_layer

    def set_output(self):
        self._output = self._prev_layer.output + self._add_layer.output


class EltwiseMultiplyLayer(Layer):

    def __init__(self, prev_layer, mult_layer):
        super().__init__(prev_layer)
        self._output_shape = self._input_shape
        self._mult_layer = mult_layer

    def set_output(self):
        self._output = self._prev_layer.output * self._mult_layer.output


class SubtractLayer(Layer):

    def __init__(self, prev_layer, subtract_val):
        super().__init__(prev_layer)
        self._output_shape = self._input_shape
        self._subtract_val = subtract_val

    def set_output(self):
        self._output = self._prev_layer.output - self._subtract_val


class FlattenLayer(Layer):

    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self._output_shape = [self._input_shape[0], np.prod(self._input_shape[1:])]

    def set_output(self):
        self._output = \
            self._prev_layer.output.flatten(2)  # flatten from the second dim


class DimShuffleLayer(Layer):

    def __init__(self, prev_layer, shuffle_pattern):
        super().__init__(prev_layer)
        self._shuffle_pattern = shuffle_pattern
        self._output_shape = list(shuffle_pattern)
        for out_dim, in_dim in enumerate(shuffle_pattern):
            if in_dim == 'x':
                self._output_shape[out_dim] = 1
            else:
                self._output_shape[out_dim] = self._input_shape[in_dim]
        self._output_shape = tuple(self._output_shape)

    def set_output(self):
        self._output = self._prev_layer.output.dimshuffle(self._shuffle_pattern)


class ReshapeLayer(Layer):

    def __init__(self, prev_layer, reshape):
        super().__init__(prev_layer)
        self._output_shape = [self._prev_layer.output_shape[0]]
        self._output_shape.extend(reshape)
        self._output_shape = tuple(self._output_shape)
        print('Reshape the prev layer to [%s]' % ','.join(str(x) for x in self._output_shape))

    def set_output(self):
        self._output = tensor.reshape(self._prev_layer.output, self._output_shape)


class ConvLayer(Layer):
    """Conv Layer
    filter_shape: [n_out_channel, n_height, n_width]

    self._input_shape: [batch_size, n_in_channel, n_height, n_width]
    """

    def __init__(self, prev_layer, filter_shape, padding=True, params=None,
                 param_type=None):
        super().__init__(prev_layer)
        self._padding = padding
        self._filter_shape = [filter_shape[0], self._input_shape[1],
                              filter_shape[1], filter_shape[2]]
        if params is None:
            self.W = Weight(self._filter_shape, is_bias=False,
                            param_type=param_type)
            self.b = Weight((filter_shape[0],), is_bias=True, mean=0.1,
                            filler='constant', param_type=param_type)
        else:
            for i, s in enumerate(self._filter_shape):
                assert (params[0].shape[i] == s)
            self.W = params[0]
            self.b = params[1]

        self.params = [self.W, self.b]

        # Define self._output_shape
        if padding and filter_shape[1] * filter_shape[2] > 1:
            self._padding = [0, 0, int((filter_shape[1] - 1) / 2), int((filter_shape[2] - 1) / 2)]
            self._output_shape = [self._input_shape[0], filter_shape[0], self._input_shape[2],
                                  self._input_shape[3]]
        else:
            self._padding = [0] * 4
            # TODO: for the 'valid' convolution mode the following is the
            # output shape. Diagnose failure
            self._output_shape = [self._input_shape[0], filter_shape[0],
                                  self._input_shape[2] - filter_shape[1] + 1,
                                  self._input_shape[3] - filter_shape[2] + 1]

    def set_output(self):
        if sum(self._padding) > 0:
            padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                        self._input_shape[0],
                                        self._input_shape[1],
                                        self._input_shape[2] + 2 * self._padding[2],
                                        self._input_shape[3] + 2 * self._padding[3])

            padded_input = tensor.set_subtensor(
                padded_input[:, :, self._padding[2]:self._padding[2] + self._input_shape[2],
                             self._padding[3]:self._padding[3] + self._input_shape[3]],
                self._prev_layer.output)

            padded_input_shape = [self._input_shape[0], self._input_shape[1],
                                  self._input_shape[2] + 2 * self._padding[2],
                                  self._input_shape[3] + 2 * self._padding[3]]
        else:
            padded_input = self._prev_layer.output
            padded_input_shape = self._input_shape

        conv_out = conv.conv2d(
            input=padded_input,
            filters=self.W.val,
            filter_shape=self._filter_shape,
            image_shape=np.asarray(
                padded_input_shape, dtype=np.int16),
            border_mode='valid')

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self._output = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')


class PoolLayer(Layer):

    def __init__(self, prev_layer, pool_size=(2, 2), padding=(1, 1)):
        super().__init__(prev_layer)
        self._pool_size = pool_size
        self._padding = padding
        img_rows = self._input_shape[2] + 2 * padding[0]
        img_cols = self._input_shape[3] + 2 * padding[1]
        out_r = (img_rows - pool_size[0]) // pool_size[0] + 1
        out_c = (img_cols - pool_size[1]) // pool_size[1] + 1
        self._output_shape = [self._input_shape[0], self._input_shape[1], out_r, out_c]

    def set_output(self):
        pooled_out = pool.pool_2d(
            input=self._prev_layer.output,
            ds=self._pool_size,
            ignore_border=True,
            padding=self._padding)
        self._output = pooled_out


class Pool3DLayer(Layer):

    def __init__(self, prev_layer, pool_size=(2, 2, 2)):
        super().__init__(prev_layer)
        self._pool_size = pool_size
        self._output_shape = (self._input_shape[0],
                              int(self._input_shape[1] / pool_size[0]),
                              self._input_shape[2],  # out channel
                              int(self._input_shape[3] / pool_size[1]),
                              int(self._input_shape[4] / pool_size[2]))

    def set_output(self):
        self._output = max_pool_3d(
                self._prev_layer.output.dimshuffle(0, 2, 1, 3, 4),
                self._pool_size, ignore_border=True).dimshuffle(0, 2, 1, 3, 4)


class Unpool3DLayer(Layer):
    """3D Unpooling layer for a convolutional network """

    def __init__(self, prev_layer, unpool_size=(2, 2, 2), padding=(0, 0, 0)):
        super().__init__(prev_layer)
        self._unpool_size = unpool_size
        self._padding = padding
        output_shape = (self._input_shape[0],  # batch
                        unpool_size[0] * self._input_shape[1] + 2 * padding[0],  # depth
                        self._input_shape[2],  # out channel
                        unpool_size[1] * self._input_shape[3] + 2 * padding[1],  # row
                        unpool_size[2] * self._input_shape[4] + 2 * padding[2])  # col
        self._output_shape = output_shape

    def set_output(self):
        output_shape = self._output_shape
        padding = self._padding
        unpool_size = self._unpool_size
        unpooled_output = tensor.alloc(0.0,  # Value to fill the tensor
                                       output_shape[0],
                                       output_shape[1] + 2 * padding[0],
                                       output_shape[2],
                                       output_shape[3] + 2 * padding[1],
                                       output_shape[4] + 2 * padding[2])

        unpooled_output = tensor.set_subtensor(unpooled_output[:, padding[0]:output_shape[
            1] + padding[0]:unpool_size[0], :, padding[1]:output_shape[3] + padding[1]:unpool_size[
                1], padding[2]:output_shape[4] + padding[2]:unpool_size[2]],
                                               self._prev_layer.output)
        self._output = unpooled_output


class Conv3DLayer(Layer):
    """3D Convolution layer"""

    def __init__(self, prev_layer, filter_shape, padding=None, params=None,
                 param_type=None):
        super().__init__(prev_layer)
        self._filter_shape = [filter_shape[0],  # out channel
                              filter_shape[1],  # time
                              self._input_shape[2],  # in channel
                              filter_shape[2],  # height
                              filter_shape[3]]  # width
        self._padding = padding

        # signals: (batch,       in channel, depth_i, row_i, column_i)
        # filters: (out channel, in channel, depth_f, row_f, column_f)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        if params is None:
            self.W = Weight(self._filter_shape, is_bias=False,
                            param_type=param_type)
            self.b = Weight((filter_shape[0],), is_bias=True, mean=0.1,
                             filler='constant', param_type=param_type)
            params = [self.W, self.b]
        else:
            assert params[0].shape == self._filter_shape
            self.W = params[0]
            self.b = params[1]

        self.params = [self.W, self.b]

        if padding is None:
            self._padding = [0, int((filter_shape[1] - 1) / 2), 0,
                             int((filter_shape[2] - 1) / 2),
                             int((filter_shape[3] - 1) / 2)]

        self._output_shape = [self._input_shape[0], self._input_shape[1],
                              filter_shape[0], self._input_shape[3],
                              self._input_shape[4]]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        if np.sum(self._padding) > 0:
            padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                        input_shape[0],
                                        input_shape[1] + 2 * padding[1],
                                        input_shape[2],
                                        input_shape[3] + 2 * padding[3],
                                        input_shape[4] + 2 * padding[4])

            padded_input = tensor.set_subtensor(
                padded_input[:, padding[1]:padding[1] + input_shape[1], :, padding[3]:padding[3] +
                             input_shape[3], padding[4]:padding[4] + input_shape[4]],
                self._prev_layer.output)
        else:
            padded_input = self._prev_layer.output

        self._output = conv3d2d.conv3d(padded_input, self.W.val) + \
            self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')


class FCConv3DLayer(Layer):
    """3D Convolution layer with FC input and hidden unit"""

    def __init__(self, prev_layer, fc_layer, filter_shape, padding=None,
                 param_type=None, params=None):
        """Prev layer is the 3D hidden layer"""
        super().__init__(prev_layer)
        self._fc_layer = fc_layer
        self._filter_shape = [filter_shape[0],  # out channel
                              filter_shape[2],  # time
                              filter_shape[1],  # in channel
                              filter_shape[3],  # height
                              filter_shape[4]]  # width
        self._padding = padding

        if padding is None:
            self._padding = [0, int((self._filter_shape[1] - 1) / 2), 0, int(
                (self._filter_shape[3] - 1) / 2), int((self._filter_shape[4] - 1) / 2)]

        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
                              self._input_shape[3], self._input_shape[4]]

        if params is None:
            self.Wh = Weight(self._filter_shape, is_bias=False,
                             param_type=param_type)

            self._Wx_shape = [self._fc_layer._output_shape[1], np.prod(self._output_shape[1:])]

            # Each 3D cell will have independent weights but for computational
            # speed, we expand the cells and compute a matrix multiplication.
            self.Wx = Weight(
                self._Wx_shape, is_bias=False,
                fan_in=self._input_shape[1], fan_out=self._output_shape[2],
                param_type=param_type)

            self.b = Weight((filter_shape[0],), is_bias=True,
                            param_type=param_type, mean=0.1, filler='constant')
            params = [self.Wh, self.Wx, self.b]
        else:
            self.Wh = params[0]
            self.Wx = params[1]
            self.b = params[2]

        self.params = [self.Wh, self.Wx, self.b]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                    input_shape[0],
                                    input_shape[1] + 2 * padding[1],
                                    input_shape[2],
                                    input_shape[3] + 2 * padding[3],
                                    input_shape[4] + 2 * padding[4])

        padded_input = tensor.set_subtensor(padded_input[:, padding[1]:padding[1] + input_shape[
            1], :, padding[3]:padding[3] + input_shape[3], padding[4]:padding[4] + input_shape[4]],
                                            self._prev_layer.output)

        fc_output = tensor.reshape(
            tensor.dot(self._fc_layer.output, self.Wx.val), self._output_shape)
        self._output = conv3d2d.conv3d(padded_input, self.Wh.val) + \
            fc_output + self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')


class Conv3DLSTMLayer(Layer):
    """Convolution 3D LSTM layer

    Unlike a standard LSTM cell witch doesn't have a spatial information,
    Convolutional 3D LSTM has limited connection that respects spatial
    configuration of LSTM cells.

    The filter_shape defines the size of neighbor that the 3D LSTM cells will consider.
    """

    def __init__(self, prev_layer, filter_shape, padding=None, params=None,
                 param_type=None):

        super().__init__(prev_layer)
        prev_layer._input_shape
        n_c = filter_shape[0]
        n_x = self._input_shape[2]
        n_neighbor_d = filter_shape[1]
        n_neighbor_h = filter_shape[2]
        n_neighbor_w = filter_shape[3]

        # Compute all gates in one convolution
        self._gate_filter_shape = [4 * n_c, 1, n_x + n_c, 1, 1]

        self._filter_shape = [filter_shape[0],  # num out hidden representation
                              filter_shape[1],  # time
                              self._input_shape[2],  # in channel
                              filter_shape[2],  # height
                              filter_shape[3]]  # width
        self._padding = padding

        # signals: (batch,       in channel, depth_i, row_i, column_i)
        # filters: (out channel, in channel, depth_f, row_f, column_f)

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        if params is None:
            self.W = Weight(self._filter_shape, is_bias=False,
                            param_type=param_type)
            self.b = Weight((filter_shape[0],), is_bias=True,
                            param_type=param_type, mean=0.1, filler='constant')
            params = [self.W, self.b]
        else:
            self.W = params[0]
            self.b = params[1]

        self.params = [self.W, self.b]

        if padding is None:
            self._padding = [0, int((filter_shape[1] - 1) / 2), 0, int((filter_shape[2] - 1) / 2),
                             int((filter_shape[3] - 1) / 2)]

        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
                              self._input_shape[3], self._input_shape[4]]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        padded_input = tensor.alloc(0.0,  # Value to fill the tensor
                                    input_shape[0],
                                    input_shape[1] + 2 * padding[1],
                                    input_shape[2],
                                    input_shape[3] + 2 * padding[3],
                                    input_shape[4] + 2 * padding[4])

        padded_input = tensor.set_subtensor(padded_input[:, padding[1]:padding[1] + input_shape[
            1], :, padding[3]:padding[3] + input_shape[3], padding[4]:padding[4] + input_shape[4]],
                                            self._prev_layer.output)

        self._output = conv3d2d.conv3d(padded_input, self.W.val) + \
            self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')


class SoftmaxWithLoss3D(object):
    """
    Softmax with loss (n_batch, n_vox, n_label, n_vox, n_vox)
    """

    def __init__(self, input, axis=2):
        self.input = input
        self.input_max = tensor.max(self.input, axis=axis, keepdims=True)
        self.exp_x = tensor.exp(self.input - self.input_max)
        self.sum_exp_x = tensor.sum(self.exp_x, axis=axis, keepdims=True)
        self.axis = axis

    def prediction(self):
        return self.exp_x / self.sum_exp_x

    def error(self, y, threshold=0.5):
        return tensor.mean(tensor.neq(tensor.ge(self.prediction(), threshold), y))

    def loss(self, y):
        """
        y must be a tensor that has the same dimensions as the input. For each
        channel, only one element is one indicating the ground truth prediction
        label.
        """
        return tensor.mean(
                tensor.sum(-y * self.input, axis=self.axis, keepdims=True) +
                self.input_max + tensor.log(self.sum_exp_x))


class ConcatLayer(Layer):

    def __init__(self, prev_layers, axis=1):
        """
        list of prev layers to concatenate
        axis to concatenate

        For tensor5, channel dimension is axis=2 (due to theano conv3d
        convention). For image, axis=1
        """
        assert (len(prev_layers) > 1)
        super().__init__(prev_layers[0])
        self._axis = axis
        self._prev_layers = prev_layers

        self._output_shape = self._input_shape.copy()
        for prev_layer in prev_layers[1:]:
            self._output_shape[axis] += prev_layer._output_shape[axis]
        print('Concat the prev layer to [%s]' % ','.join(str(x) for x in self._output_shape))

    def set_output(self):
        self._output = tensor.concatenate([x.output for x in self._prev_layers], axis=self._axis)


class LeakyReLU(Layer):

    def __init__(self, prev_layer, leakiness=0.01):
        super().__init__(prev_layer)
        self._leakiness = leakiness
        self._output_shape = self._input_shape

    def set_output(self):
        input_ = self._prev_layer.output
        if self._leakiness:
            self._output = tensor.maximum(input_, self._leakiness * input_)
        else:
            self._output = tensor.maximum(input_, 0.)


class SigmoidLayer(Layer):

    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = sigmoid(self._prev_layer.output)


class TanhLayer(Layer):

    def __init__(self, prev_layer):
        super().__init__(prev_layer)

    def set_output(self):
        self._output = tensor.tanh(self._prev_layer.output)


class DifferentiableStepLayer(Layer):

    def __init__(self, prev_layer, backprop='linear'):
        self.op = DifferentiableStepOp(backprop)
        super().__init__(prev_layer)
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = self.op(self._prev_layer.output)


class InstanceNoiseLayer(Layer):

    def __init__(self, prev_layer, std=1.):
        super().__init__(prev_layer)
        self._std = std
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = self._prev_layer.output + get_rng().normal(
                self._input_shape, std=self._std)


class RaytracingLayer(Layer):
    def __init__(self, prev_layer, cams, img_w, img_h, pad_x, pad_y,
                 mode='maxpool'):
        super().__init__(prev_layer)
        self._input = self._prev_layer.output
        if len(self._input_shape) == 5:
            self._input_shape = (self._input_shape[0],  # batch
                                 self._input_shape[1],  # v
                                 self._input_shape[3],  # v
                                 self._input_shape[4],  # v
                                 self._input_shape[2],)  # dim
            self._input = self._input.dimshuffle(0, 1, 3, 4, 2)
        elif len(self._input_shape) == 4:
            self._input_shape = (self._input_shape[0],  # batch
                                 self._input_shape[1],  # v
                                 self._input_shape[2],  # v
                                 self._input_shape[3],  # v
                                 1,)  # dim
            self._input = self._input.dimshuffle(0, 1, 2, 3, 'x')
        else:
            raise ValueError('Unresolvable input shape for raytracing layer.')
        assert (self._input_shape[1] ==
                self._input_shape[2] ==
                self._input_shape[3])
        voxel_d = self._input_shape[1]
        feat_d = self._input_shape[4]
        self._output_shape = [None,  # Number of views. Variable.
                              self._input_shape[0],  # batch
                              feat_d,  # dim
                              img_h,
                              img_w]
        self.camera_params = cams
        self.img_w = img_w
        self.img_h = img_h
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.feat_d = feat_d
        self.op = RaytraceVoxelOp(img_w + pad_x, img_h + pad_y, voxel_d, feat_d,
                                  mode)

    def get_blender_proj(self, camera):
        deg2rad = lambda angle: (angle / 180.) * np.pi
        sa = tensor.sin(deg2rad(-camera[0]))
        ca = tensor.cos(deg2rad(-camera[0]))
        se = tensor.sin(deg2rad(-camera[1]))
        ce = tensor.cos(deg2rad(-camera[1]))
        R_world2obj = tensor.eye(3)
        R_world2obj = tensor.set_subtensor(R_world2obj[0, 0], ca * ce)
        R_world2obj = tensor.set_subtensor(R_world2obj[0, 1], sa * ce)
        R_world2obj = tensor.set_subtensor(R_world2obj[0, 2], -se)
        R_world2obj = tensor.set_subtensor(R_world2obj[1, 0], -sa)
        R_world2obj = tensor.set_subtensor(R_world2obj[1, 1], ca)
        R_world2obj = tensor.set_subtensor(R_world2obj[2, 0], ca * se)
        R_world2obj = tensor.set_subtensor(R_world2obj[2, 1], sa * se)
        R_world2obj = tensor.set_subtensor(R_world2obj[2, 2], ce)
        R_obj2cam = np.array(
                ((1.910685676922942e-15, 4.371138828673793e-08, 1.0),
                 (1.0, -4.371138828673793e-08, -0.0),
                 (4.371138828673793e-08, 1.0, -4.371138828673793e-08))).T
        R_world2cam = tensor.dot(R_obj2cam, R_world2obj)
        cam_location = tensor.zeros((3, 1))
        cam_location = tensor.set_subtensor(cam_location[0, 0], camera[2] * 1.75)
        T_world2cam = -1 * tensor.dot(R_obj2cam, cam_location)
        R_camfix = np.array(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
        R_world2cam = tensor.dot(R_camfix, R_world2cam)
        T_world2cam = tensor.dot(R_camfix, T_world2cam)
        RT = tensor.concatenate([R_world2cam, T_world2cam], axis=1)
        return RT

    def get_binvox_proj(self, binvox_params):
        """Calculate 4x4 projection matrix from voxel to obj coordinate"""
        # Calculate rotation and translation matrices.
        # Step 1: Voxel coordinate to binvox coordinate.
        S_vox2bin = tensor.eye(4) * (1 / (binvox_params[0] - 1))
        S_vox2bin = tensor.set_subtensor(S_vox2bin[3, 3], 1.)

        # Step 2: Binvox coordinate to obj coordinate.
        voxel_t = tensor.min(binvox_params[1:4])
        b_rot = np.array(((1.0, 0.0, 0.0, 0.0),
                          (0.0, 7.549790126404332e-08, -1.0, 0.0),
                          (0.0, 1.0, 7.549790126404332e-08, 0.0),
                          (0.0, 0.0, 0.0, 1.0)))
        b_t = tensor.eye(4)
        b_t = tensor.set_subtensor(b_t[:3, 3], voxel_t)
        b_s = tensor.eye(4) * binvox_params[4]
        b_s = tensor.set_subtensor(b_s[3, 3], 1.)
        RST_bin2obj = tensor.dot(tensor.dot(b_rot, b_t), b_s)
        return tensor.dot(RST_bin2obj, S_vox2bin)

    def get_camloc(self, camera_params):
        RT = self.get_blender_proj(camera_params[:3])
        W2B = self.get_binvox_proj(camera_params[3:-3])
        invloc = -1 * tensor.dot(RT[:, :3].T, RT[:, 3].reshape((3, 1)))
        invloc_c = tensor.concatenate([invloc.T, tensor.ones((1, 1))], axis=1)
        camloc = tensor.dot(invloc_c, tensor.nlinalg.matrix_inverse(W2B).T)
        return camloc[0, :3] / camloc[0, 3]

    def get_raydirs(self, camera_params):
        img_w = self.img_w + self.pad_x
        img_h = self.img_h + self.pad_y
        K = np.array(((35. * img_w / 32., 0, img_w / 2.),
                     (0, 35. * img_h / 32., img_h / 2.),
                     (0, 0, 1)))
        RT = self.get_blender_proj(camera_params[:3])
        W2B = self.get_binvox_proj(camera_params[3:-3])
        camloc = self.get_camloc(camera_params)
        pixloc = np.array(list(itertools.product(range(img_h), range(img_w), (1,))))
        pixloc = tensor.dot(pixloc, (tensor.dot(tensor.dot(tensor.nlinalg.matrix_inverse(W2B), tensor.nlinalg.pinv(RT)), tensor.nlinalg.matrix_inverse(K))).T)
        pixloc = pixloc[:, :3] / tensor.tile(pixloc[:, 3].dimshuffle(0, 'x'), (1, 3))
        return tensor.tile(camloc.dimshuffle('x', 0), (pixloc.shape[0], 1)) - pixloc

    def data_augmentation(self, img, camera_params):
        cr = camera_params[8].astype('int32')
        cc = camera_params[9].astype('int32')
        img_sub = img[cr:cr + self.img_h, cc:cc + self.img_w]
        return theano.ifelse.ifelse(tensor.gt(camera_params[10], 0.5),
                                    img_sub[:, ::-1], img_sub)

    def get_camloc_real(self, camera_params):
        return camera_params[3:6]

    def get_raydirs_real(self, camera_params):
        return camera_params[6:]

    def data_augmentation_real(self, img, camera_params):
        cr = camera_params[0].astype('int32')
        cc = camera_params[1].astype('int32')
        img_sub = img[cr:cr + self.img_h, cc:cc + self.img_w]
        return theano.ifelse.ifelse(tensor.gt(camera_params[2], 0.5),
                                    img_sub[:, ::-1], img_sub)

    def set_output(self):
        cshape = self.camera_params.shape  # (c, batch, 11)
        voxel = self._input
        voxel_tiled = tensor.tile(voxel, (cshape[0], 1, 1, 1, 1))
        cams = tensor.reshape(self.camera_params,
                              (cshape[0] * cshape[1], cshape[2]))
        camlocs = theano.map(self.get_camloc, cams)[0].astype('float32')
        raydirs = theano.map(self.get_raydirs, cams)[0].astype('float32')
        rendered = self.op(voxel_tiled, camlocs, raydirs)
        rendered_sub = theano.map(self.data_augmentation, [rendered, cams])[0]
        output_shape = (cshape[0], cshape[1], self.img_h, self.img_w,
                        self.feat_d)
        self._output = tensor.reshape(rendered_sub, output_shape)
        self._output = self._output.dimshuffle(0, 1, 4, 2, 3)


class ComplementLayer(Layer):
    """ Compute 1 - input_layer.output """

    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = tensor.ones_like(self._prev_layer.output) - self._prev_layer.output
