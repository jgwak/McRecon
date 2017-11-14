import numpy as np
import theano
import theano.misc.pycuda_init
from pycuda.compiler import SourceModule
import theano.sandbox.cuda as cuda

import lib.get_projection as get_projection

POOL_MODES = ('maxpool', 'meanpool')

RAYTRACE_CUDA = """
#include <float.h>
#define IDX2D(i, j, dj) (dj * i + j)
#define IDX3D(i, j, k, dj, dk) (IDX2D(IDX2D(i, j, dj), k, dk))
#define IDX4D(i, j, k, l, dj, dk, dl) (IDX2D(IDX3D(i, j, k, dj, dk), l, dl))
#define IDX_FEAT(i, j, k, l, dv, df) IDX4D(i, j, k, l, dv, dv, df)
#define MAX_REND_IDX(voxel_d) (voxel_d * 3)

__device__ bool intersect(float* camloc, float* raydir, int bbx[3], int size)
{
    float invdir[3];
    int sign[3];
    float bbxs[2][3];
    for(int i = 0; i < 3; i++)
    {
        invdir[i] = 1. / raydir[i];
        sign[i] = invdir[i] < 0;
        bbxs[0][i] = (float)bbx[i];
        bbxs[1][i] = (float)bbx[i] + size;
    }

    float tmin = (bbxs[sign[0]][0] - camloc[0]) * invdir[0];
    float tmax = (bbxs[1 - sign[0]][0] - camloc[0]) * invdir[0];
    float tymin = (bbxs[sign[1]][1] - camloc[1]) * invdir[1];
    float tymax = (bbxs[1 - sign[1]][1] - camloc[1]) * invdir[1];

    if ((tmin > tymax) || (tymin > tmax))
        return false;
    if (tymin > tmin)
        tmin = tymin;
    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (bbxs[sign[2]][2] - camloc[2]) * invdir[2];
    float tzmax = (bbxs[1 - sign[2]][2] - camloc[2]) * invdir[2];

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;
    return true;
}


__device__ void raytrace_divide(float* camloc, float* raydirs, int* rendering,
        int* rend_idx, int voxel_d, int img_h, int img_w, int bbox[3], int size,
        int img_x, int img_y)
{
    float* raydir = raydirs + IDX3D(img_x, img_y, 0, img_h, 3);
    if (intersect(camloc, raydir, bbox, size))
    {
        if (size == 1) {
            int curr_idx = atomicAdd(rend_idx + IDX2D(img_y, img_x, img_w), 1);
            int max_idx = MAX_REND_IDX(voxel_d);
            int voxel_i = IDX3D(bbox[0], bbox[1], bbox[2], voxel_d, voxel_d);
            rendering[IDX3D(img_y, img_x, curr_idx, img_w, max_idx)] = voxel_i;
        } else {
            size /= 2;
            for (int i = 0; i < 2; i++){
                for (int j = 0; j < 2; j++) {
                    for(int k = 0; k < 2; k++){
                        int bbox_new[3] = {bbox[0] + i * size,
                                           bbox[1] + j * size,
                                           bbox[2] + k * size};
                        raytrace_divide(camloc, raydirs, rendering, rend_idx,
                                voxel_d, img_h, img_w, bbox_new, size, img_x,
                                img_y);
                    }
                }
            }
        }
    }
}


__global__ void raytrace(float* camloc, float* raydirs, int* rendering,
        int* rend_idx, int voxel_d, int img_h, int img_w, int batch_size)
{
    int img_y = threadIdx.x + blockIdx.x * blockDim.x;
    int img_x = threadIdx.y + blockIdx.y * blockDim.y;
    int batch_idx = threadIdx.z + blockIdx.z * blockDim.z;
    int img_size = img_w * img_h;
    if (batch_idx < batch_size && img_y < img_h && img_x < img_w)
    {
        int bbox[3] = {0, 0, 0};
        int size = voxel_d;
        raytrace_divide(camloc + batch_idx * 3,
                        raydirs + batch_idx * img_size * 3,
                        rendering + batch_idx * img_size * MAX_REND_IDX(voxel_d),
                        rend_idx + batch_idx * img_size, voxel_d, img_h, img_w,
                        bbox, size, img_x, img_y);
    }
}


__global__ void maxpool(float* feat, int* rendering, int* rend_idx,
        float* output, int voxel_d, int img_h, int img_w, int feat_d, int batch_size)
{
    int num_voxel = voxel_d * voxel_d * voxel_d;
    int img_size = img_w * img_h;
    int batch_idx = (threadIdx.x + blockIdx.x * blockDim.x) / feat_d;
    int feat_idx = (threadIdx.x + blockIdx.x * blockDim.x) % feat_d;
    int img_y = threadIdx.y + blockIdx.y * blockDim.y;
    int img_x = threadIdx.z + blockIdx.z * blockDim.z;
    if (batch_idx < batch_size && feat_idx < feat_d && img_y < img_h && img_x < img_w)
    {
        feat += batch_idx * num_voxel * feat_d;
        rendering += batch_idx * img_size * MAX_REND_IDX(voxel_d);
        rend_idx += batch_idx * img_size;
        output += batch_idx * img_size * feat_d;
        int num_hit = rend_idx[IDX2D(img_y, img_x, img_w)];
        if (num_hit > 0) {
            float max_val = -FLT_MAX;
            for (int i = 0; i < num_hit; i++)
            {
                int voxel_i = rendering[
                        IDX3D(img_y, img_x, i, img_w, MAX_REND_IDX(voxel_d))];
                float curr_val = feat[IDX2D(voxel_i, feat_idx, feat_d)];
                if (curr_val > max_val)
                {
                    max_val = curr_val;
                }
            }
            output[IDX3D(img_y, img_x, feat_idx, img_w, feat_d)] = max_val;
        } else {
            output[IDX3D(img_y, img_x, feat_idx, img_w, feat_d)] = 0;
        }
    }
}


__global__ void maxpool_grad(float* feat, int* rendering, int* rend_idx,
        float* doutput, float* dfeat, int voxel_d, int img_h, int img_w,
        int feat_d, int batch_size)
{
    int num_voxel = voxel_d * voxel_d * voxel_d;
    int img_size = img_w * img_h;
    int batch_idx = (threadIdx.x + blockIdx.x * blockDim.x) / feat_d;
    int feat_idx = (threadIdx.x + blockIdx.x * blockDim.x) % feat_d;
    int img_y = threadIdx.y + blockIdx.y * blockDim.y;
    int img_x = threadIdx.z + blockIdx.z * blockDim.z;
    if (batch_idx < batch_size && feat_idx < feat_d && img_y < img_h && img_x < img_w)
    {
        feat += batch_idx * num_voxel * feat_d;
        rendering += batch_idx * img_size * MAX_REND_IDX(voxel_d);
        rend_idx += batch_idx * img_size;
        doutput += batch_idx * img_size * feat_d;
        dfeat += batch_idx * num_voxel * feat_d;
        int num_hit = rend_idx[IDX2D(img_y, img_x, img_w)];
        if (num_hit > 0) {
            int max_voxel_i = -1;
            float max_val = -FLT_MAX;
            for (int i = 0; i < num_hit; i++)
            {
                int voxel_i = rendering[
                        IDX3D(img_y, img_x, i, img_w, MAX_REND_IDX(voxel_d))];
                float curr_val = feat[IDX2D(voxel_i, feat_idx, feat_d)];
                if (curr_val > max_val)
                {
                    max_val = curr_val;
                    max_voxel_i = voxel_i;
                }
            }
            atomicAdd(dfeat + IDX2D(max_voxel_i, feat_idx, feat_d),
                    doutput[IDX3D(img_y, img_x, feat_idx, img_w, feat_d)]);
        }
    }
}


__global__ void meanpool(float* feat, int* rendering, int* rend_idx,
        float* output, int voxel_d, int img_h, int img_w, int feat_d, int batch_size)
{
    int num_voxel = voxel_d * voxel_d * voxel_d;
    int img_size = img_w * img_h;
    int batch_idx = (threadIdx.x + blockIdx.x * blockDim.x) / feat_d;
    int feat_idx = (threadIdx.x + blockIdx.x * blockDim.x) % feat_d;
    int img_y = threadIdx.y + blockIdx.y * blockDim.y;
    int img_x = threadIdx.z + blockIdx.z * blockDim.z;
    if (batch_idx < batch_size && feat_idx < feat_d && img_y < img_h && img_x < img_w)
    {
        feat += batch_idx * num_voxel * feat_d;
        rendering += batch_idx * img_size * MAX_REND_IDX(voxel_d);
        rend_idx += batch_idx * img_size;
        output += batch_idx * img_size * feat_d;
        int num_hit = rend_idx[IDX2D(img_y, img_x, img_w)];
        if (num_hit > 0) {
            float curr_sum = 0;
            for (int i = 0; i < num_hit; i++)
            {
                int voxel_i = rendering[
                        IDX3D(img_y, img_x, i, img_w, MAX_REND_IDX(voxel_d))];
                curr_sum += feat[IDX2D(voxel_i, feat_idx, feat_d)] / (float)num_hit;
            }
            output[IDX3D(img_y, img_x, feat_idx, img_w, feat_d)] = curr_sum;
        } else {
            output[IDX3D(img_y, img_x, feat_idx, img_w, feat_d)] = 0;
        }
    }
}


__global__ void meanpool_grad(float* feat, int* rendering, int* rend_idx,
        float* doutput, float* dfeat, int voxel_d, int img_h, int img_w,
        int feat_d, int batch_size)
{
    int num_voxel = voxel_d * voxel_d * voxel_d;
    int img_size = img_w * img_h;
    int batch_idx = (threadIdx.x + blockIdx.x * blockDim.x) / feat_d;
    int feat_idx = (threadIdx.x + blockIdx.x * blockDim.x) % feat_d;
    int img_y = threadIdx.y + blockIdx.y * blockDim.y;
    int img_x = threadIdx.z + blockIdx.z * blockDim.z;
    if (batch_idx < batch_size && feat_idx < feat_d && img_y < img_h && img_x < img_w)
    {
        feat += batch_idx * num_voxel * feat_d;
        rendering += batch_idx * img_size * MAX_REND_IDX(voxel_d);
        rend_idx += batch_idx * img_size;
        doutput += batch_idx * img_size * feat_d;
        dfeat += batch_idx * num_voxel * feat_d;
        int num_hit = rend_idx[IDX2D(img_y, img_x, img_w)];
        if (num_hit > 0) {
            for (int i = 0; i < num_hit; i++)
            {
                int voxel_i = rendering[
                        IDX3D(img_y, img_x, i, img_w, MAX_REND_IDX(voxel_d))];
                atomicAdd(dfeat + IDX2D(voxel_i, feat_idx, feat_d),
                        doutput[IDX3D(img_y, img_x, feat_idx, img_w, feat_d)] /
                        (float)num_hit);
            }
        }
    }
}"""

class RaytraceVoxelOp(theano.Op):
    __props__ = ('img_w', 'img_h', 'voxel_d', 'feat_d', 'mode')

    def __init__(self, img_w, img_h, voxel_d, feat_d, mode):
        self.img_w = img_w
        self.img_h = img_h
        self.voxel_d = voxel_d
        self.feat_d = feat_d
        if mode not in POOL_MODES:
            raise ValueError('Invalid raytrace mode %s.' % mode)
        self.mode = mode
        super(RaytraceVoxelOp, self).__init__()

    def make_node(self, feat, camlocs, raydirs):
        feat = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(feat))
        camlocs = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(camlocs))
        raydirs = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(raydirs))
        output = cuda.type.CudaNdarrayType((False,)*4, dtype='float32')()
        return theano.Apply(self, [feat, camlocs, raydirs], [output])

    def make_thunk(self, node, storage_map, _, _2):
        raytrace_mod = SourceModule(RAYTRACE_CUDA)
        raytrace_fct = raytrace_mod.get_function('raytrace')
        pool_fct = raytrace_mod.get_function(self.mode)
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        def thunk():
            batch_size = inputs[0][0].shape[0]
            output_shape = (batch_size, self.img_h, self.img_w, self.feat_d)
            max_rend_idx = self.voxel_d * 3
            num_voxel = self.voxel_d ** 3
            img_size = self.img_w * self.img_h
            if outputs[0][0] is None or outputs[0][0].shape != output_shape:
                outputs[0][0] = cuda.CudaNdarray.zeros(output_shape)
            rendering = cuda.CudaNdarray.zeros((batch_size, num_voxel,
                                                max_rend_idx))
            rend_idx = cuda.CudaNdarray.zeros((batch_size, img_size))
            block = (8, 8, 8)
            rend_grid = (int(np.ceil(self.img_h / 8.)),
                         int(np.ceil(self.img_w / 8.)),
                         int(np.ceil(batch_size / 8.)))
            feat_grid = (int(np.ceil(self.feat_d * batch_size / 8.)),
                         int(np.ceil(self.img_h / 8.)),
                         int(np.ceil(self.img_w / 8.)))
            raytrace_fct(inputs[1][0], inputs[2][0], rendering, rend_idx,
                    np.intc(self.voxel_d), np.intc(self.img_h),
                    np.intc(self.img_w), np.intc(batch_size),
                    block=block, grid=rend_grid)
            pool_fct(inputs[0][0], rendering, rend_idx, outputs[0][0],
                    np.intc(self.voxel_d), np.intc(self.img_h),
                    np.intc(self.img_w), np.intc(self.feat_d),
                    np.intc(batch_size), block=block, grid=feat_grid)
        thunk.lazy = False
        return thunk

    def grad(self, inputs, output_grads):
        return [RaytraceVoxelGradOp(self.img_w, self.img_h, self.voxel_d,
                self.feat_d, self.mode)(*inputs, output_grads[0]),
                theano.gradient.grad_not_implemented(self, 1, inputs[1]),
                theano.gradient.grad_not_implemented(self, 2, inputs[2])]

    def R_op(self, inputs, eval_points):
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)


class RaytraceVoxelGradOp(theano.Op):
    __props__ = ('img_w', 'img_h', 'voxel_d', 'feat_d', 'mode')

    def __init__(self, img_w, img_h, voxel_d, feat_d, mode):
        self.img_w = img_w
        self.img_h = img_h
        self.voxel_d = voxel_d
        self.feat_d = feat_d
        if mode not in POOL_MODES:
            raise ValueError('Invalid raytrace mode %s.' % mode)
        self.mode = mode
        super(RaytraceVoxelGradOp, self).__init__()

    def make_node(self, feat, camlocs, raydirs, dproj):
        feat = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(feat))
        camlocs = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(camlocs))
        raydirs = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(raydirs))
        dproj = cuda.basic_ops.gpu_contiguous(
                cuda.basic_ops.as_cuda_ndarray_variable(dproj))
        dfeat = cuda.type.CudaNdarrayType((False,)*5, dtype='float32')()
        return theano.Apply(self, [feat, camlocs, raydirs, dproj],
                            [dfeat])

    def make_thunk(self, node, storage_map, _, _2):
        raytrace_mod = SourceModule(RAYTRACE_CUDA)
        raytrace_fct = raytrace_mod.get_function('raytrace')
        pool_grad_fct = raytrace_mod.get_function(self.mode + '_grad')
        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]
        def thunk():
            batch_size = inputs[0][0].shape[0]
            output_shape = (batch_size, self.voxel_d, self.voxel_d,
                            self.voxel_d, self.feat_d)
            max_rend_idx = self.voxel_d * 3
            num_voxel = self.voxel_d ** 3
            img_size = self.img_w * self.img_h
            outputs[0][0] = cuda.CudaNdarray.zeros(output_shape)
            rendering = cuda.CudaNdarray.zeros((batch_size, num_voxel,
                                                max_rend_idx))
            rend_idx = cuda.CudaNdarray.zeros((batch_size, img_size))
            block = (8, 8, 8)
            rend_grid = (int(np.ceil(self.img_h / 8.)),
                         int(np.ceil(self.img_w / 8.)),
                         int(np.ceil(batch_size / 8.)))
            feat_grid = (int(np.ceil(self.feat_d * batch_size / 8.)),
                         int(np.ceil(self.img_h / 8.)),
                         int(np.ceil(self.img_w / 8.)))
            raytrace_fct(inputs[1][0], inputs[2][0], rendering, rend_idx,
                    np.intc(self.voxel_d), np.intc(self.img_h),
                    np.intc(self.img_w), np.intc(batch_size),
                    block=block, grid=rend_grid)
            pool_grad_fct(inputs[0][0], rendering, rend_idx, inputs[3][0],
                    outputs[0][0], np.intc(self.voxel_d), np.intc(self.img_h),
                    np.intc(self.img_w), np.intc(self.feat_d),
                    np.intc(batch_size), block=block, grid=feat_grid)
        thunk.lazy = False
        return thunk
