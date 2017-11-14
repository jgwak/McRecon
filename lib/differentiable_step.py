import numpy as np
import theano
import theano.misc.pycuda_init
from pycuda.compiler import SourceModule
import theano.sandbox.cuda as cuda

BACKPROP_MODES = ('linear', 'ignore')

class DifferentiableStepOp(theano.Op):
    __props__ = ('backprop',)

    def __init__(self, backprop):
        if backprop not in BACKPROP_MODES:
            raise ValueError('Invalid backprop mode %s.' % backprop)
        self.backprop = backprop
        super(DifferentiableStepOp, self).__init__()

    def make_node(self, inp):
        inp = cuda.basic_ops.gpu_contiguous(
               cuda.basic_ops.as_cuda_ndarray_variable(inp))
        return theano.Apply(self, [inp], [inp.type()])

    def make_thunk(self, node, storage_map, _, _2):
        mod = SourceModule("""
        __global__ void step_fct(float* input, float* output, int size) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if(i < size){
                if (input[i] > 0.5) {
                    output[i] = 1.;
                }
                else {
                    output[i] = 0.;
                }
            }
        }
        """)
        step_fct = mod.get_function("step_fct")
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        def thunk():
            z = outputs[0]
            if z[0] is None or z[0].shape != inputs[0][0].shape:
                z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
            grid = (int(np.ceil(inputs[0][0].size / 512.)),1)
            step_fct(inputs[0][0], z[0], np.intc(inputs[0][0].size),
                    block=(512, 1, 1), grid=grid)
        thunk.lazy = False
        return thunk

    def grad(self, inputs, output_grads):
        return [DifferentiableStepGradOp(self.backprop)(
                *inputs, output_grads[0])]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


class DifferentiableStepGradOp(theano.Op):
    __props__ = ('backprop',)

    def __init__(self, backprop):
        if not backprop in BACKPROP_MODES:
            raise ValueError('Invalid backprop mode %s.' % backprop)
        self.backprop = backprop
        super(DifferentiableStepGradOp, self).__init__()

    def make_node(self, inp, dout):
        inp = cuda.basic_ops.gpu_contiguous(
               cuda.basic_ops.as_cuda_ndarray_variable(inp))
        dout = cuda.basic_ops.gpu_contiguous(
               cuda.basic_ops.as_cuda_ndarray_variable(dout))
        return theano.Apply(self, [inp, dout], [inp.type()])

    def make_thunk(self, node, storage_map, _, _2):
        mod = SourceModule("""
        __global__ void linear(float* input, float* doutput,
                float* dinput, int size) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                if (0.4 < input[i] && input[i] < 0.6) {
                    dinput[i] = 5 * doutput[i];
                }
            }
        }
        __global__ void ignore(float* input, float* doutput,
                float* dinput, int size) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < size) {
                dinput[i] = doutput[i];
            }
        }
        """)
        step_fct_grad = mod.get_function(self.backprop)
        print('done')
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        def thunk():
            z = outputs[0]
            if z[0] is None or z[0].shape != inputs[0][0].shape:
                z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
            grid = (int(np.ceil(inputs[0][0].size / 512.)), 1)
            step_fct_grad(inputs[0][0], inputs[1][0], z[0],
                    np.intc(inputs[0][0].size), block=(512, 1, 1), grid=grid)
        thunk.lazy = False
        return thunk
