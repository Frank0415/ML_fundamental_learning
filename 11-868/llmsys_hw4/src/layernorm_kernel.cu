#include "includes/block_reduce.h"
#include "includes/cuda_util.h"
#include "includes/kernels.h"

#include <algorithm>
#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32

/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size`

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullptr
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {

  /// BEGIN ASSIGN4_2_1
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for
  // speedup

  // Step 1
  float l_sum = 0;
  float l_squared = 0;
  const float4 *inp_f4 =
      reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_squared += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }
  // Step 2
  blockReduce<ReduceType::kSum, 1>(&l_sum);
  blockReduce<ReduceType::kSum, 1>(&l_squared);
  float mean = l_sum / (hidden_size * 4);
  float inv_std =
      rsqrtf(l_squared / (hidden_size * 4) - mean * mean + LN_EPSILON);
  // Only thread 0 writes mean and variance to global memory
  if (threadIdx.x == 0) {
    vars[blockIdx.x] = l_squared / (hidden_size * 4) - mean * mean + LN_EPSILON;
    if (means != nullptr) {
      means[blockIdx.x] = mean;
    }
  }
  // Step 3
  float4 *ln_res_f4 =
      reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);

  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 x = inp_f4[idx];
    float4 s = scale_f4[idx];
    float4 b = bias_f4[idx];

    float4 result;
    result.x = (x.x - mean) * inv_std * s.x + b.x;
    result.y = (x.y - mean) * inv_std * s.y + b.y;
    result.z = (x.z - mean) * inv_std * s.z + b.z;
    result.w = (x.w - mean) * inv_std * s.w + b.w;

    ln_res_f4[idx] = result;
  }
  // assert(false && "Not Implemented");
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                      const float *inp, const float *scale, const float *bias,
                      int batch_size, int hidden_dim, cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;

  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);
}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backward kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void
ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad, const T *out_grad,
                        const T *inp, const T *gamma, const T *betta,
                        const T *vars, const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1: Compute partial gradients by looping across rows
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  float dbetta_partial = 0.0f;
  float dgamma_partial = 0.0f;

  for (int r = threadIdx.y; r < rows; r += TILE_DIM) {
    float inv_std = rsqrtf(vars[r] + LN_EPSILON);
    float xhat = (inp[r * width + col] - means[r]) * inv_std;
    float dy = out_grad[r * width + col];
    dbetta_partial += dy;
    dgamma_partial += dy * xhat;
  }

  // Step 2: Store the partial gradients in shared memory
  betta_buffer[threadIdx.y][threadIdx.x] = dbetta_partial;
  gamma_buffer[threadIdx.y][threadIdx.x] = dgamma_partial;
  __syncthreads();

  // Step 3: Reduce sum across threadIdx.y for each column
  // Tree reduction in shared memory along y dimension
  for (int stride = TILE_DIM / 2; stride > 0; stride >>= 1) {
    if (threadIdx.y < stride) {
      betta_buffer[threadIdx.y][threadIdx.x] +=
          betta_buffer[threadIdx.y + stride][threadIdx.x];
      gamma_buffer[threadIdx.y][threadIdx.x] +=
          gamma_buffer[threadIdx.y + stride][threadIdx.x];
    }
    __syncthreads();
  }

  // Step 4: Write final result to global memory
  if (threadIdx.y == 0 && col < width) {
    betta_grad[col] = betta_buffer[0][threadIdx.x];
    gamma_grad[col] = gamma_buffer[0][threadIdx.x];
  }
  /// END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backward kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {

  /// BEGIN ASSIGN4_2_2

  // Step 1 & 2: Compute dxhat and xhat with float4, compute sums
  const float4 *out_grad_f4 =
      reinterpret_cast<const float4 *>(out_grad) + blockIdx.x * hidden_dim;
  const float4 *inp_f4 =
      reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_dim;
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);

  float mean = means[blockIdx.x];
  float inv_std = rsqrtf(vars[blockIdx.x] + LN_EPSILON);

  float l_sum_dxhat = 0;
  float l_sum_dxhat_xhat = 0;

  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 dy = out_grad_f4[idx];
    float4 g = gamma_f4[idx];
    float4 x = inp_f4[idx];

    float4 dxhat;
    dxhat.x = dy.x * g.x;
    dxhat.y = dy.y * g.y;
    dxhat.z = dy.z * g.z;
    dxhat.w = dy.w * g.w;

    float4 xhat;
    xhat.x = (x.x - mean) * inv_std;
    xhat.y = (x.y - mean) * inv_std;
    xhat.z = (x.z - mean) * inv_std;
    xhat.w = (x.w - mean) * inv_std;

    l_sum_dxhat += dxhat.x + dxhat.y + dxhat.z + dxhat.w;
    l_sum_dxhat_xhat +=
        dxhat.x * xhat.x + dxhat.y * xhat.y +
        dxhat.z * xhat.z + dxhat.w * xhat.w;
  }

  // Step 3: Reduce sums across threads
  blockReduce<ReduceType::kSum, 1>(&l_sum_dxhat);
  blockReduce<ReduceType::kSum, 1>(&l_sum_dxhat_xhat);

  // Step 4: Compute final gradient with a second pass
  float D = (float)(hidden_dim * 4);
  float4 *inp_grad_f4 =
      reinterpret_cast<float4 *>(inp_grad) + blockIdx.x * hidden_dim;

  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 dy = out_grad_f4[idx];
    float4 g = gamma_f4[idx];
    float4 x = inp_f4[idx];

    float4 dxhat;
    dxhat.x = dy.x * g.x;
    dxhat.y = dy.y * g.y;
    dxhat.z = dy.z * g.z;
    dxhat.w = dy.w * g.w;

    float4 xhat;
    xhat.x = (x.x - mean) * inv_std;
    xhat.y = (x.y - mean) * inv_std;
    xhat.z = (x.z - mean) * inv_std;
    xhat.w = (x.w - mean) * inv_std;

    float4 dinp;
    dinp.x = (dxhat.x - (l_sum_dxhat + xhat.x * l_sum_dxhat_xhat) / D) *
             inv_std;
    dinp.y = (dxhat.y - (l_sum_dxhat + xhat.y * l_sum_dxhat_xhat) / D) *
             inv_std;
    dinp.z = (dxhat.z - (l_sum_dxhat + xhat.z * l_sum_dxhat_xhat) / D) *
             inv_std;
    dinp.w = (dxhat.w - (l_sum_dxhat + xhat.w * l_sum_dxhat_xhat) / D) *
             inv_std;

    inp_grad_f4[idx] = dinp;
  }

  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp,
                         const float *gamma, const float *betta,
                         const float *vars, const float *means, int batch_size,
                         int hidden_dim, cudaStream_t stream_1,
                         cudaStream_t stream_2) {

  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp,
      *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the
  // specified dimension, rounds it up.
  dim3 grid_dim((hidden_dim + TILE_DIM - 1) / TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means,
      hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}
}
} // namespace cuda
} // namespace lightseq
