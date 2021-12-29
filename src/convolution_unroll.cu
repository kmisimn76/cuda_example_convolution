/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling
 * approach. It has been written for clarity of exposition to illustrate various
 * CUDA programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication. See also: V. Volkov and
 * J. Demmel, "Benchmarking GPUs to tune dense linear algebra," in Proc. 2008
 * ACM/IEEE Conf. on Supercomputing (SC '08), Piscataway, NJ: IEEE Press, 2008,
 * pp. Art. 31:1-11.
 */

// System includes
#include <assert.h>
#include <stdio.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
//#include <helper_cuda.h>
//#include <helper_functions.h>


// apparently OpenCL only likes arrays ...
#define WH_in 114 //input width/height: WH+RS-1
#define WH 112	//output width/height
#define RS 3	//filter size
#define C 4		//input channel
#define K 4		//output channel

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
__global__ void ConvCUDA(
		float* output,
		const float* input,
		const float* weight,
		const int _WH_in,
		const int _WH,
		const int _RS,
		const int _C,
		const int _K) {

	int w = threadIdx.x + blockIdx.x * blockDim.x;
	int h = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
	int idx_output = k*_WH*_WH + h*_WH + w;

	// Original
	/*output[idx_output] = 0;
	for (int r=0;r<_RS;r++) {
		for (int s=0;s<_RS;s++) {
			for (int c=0;c<_C;c++) {
				int idx_input = c*_WH_in*_WH_in + (h+r)*_WH_in + (w+s);
				int idx_weight = k*_C*_RS*_RS + c*_RS*_RS + r*_RS + s;
				output[idx_output] += input[idx_input] * weight[idx_weight];
			}
		}
	}*/

	// Loop unrolling version
	/*output[idx_output] = 0;
	for (int r=0;r<_RS;r++) {
		for (int s=0;s<_RS;s++) {
			for (int c=0;c<_C;c+=4) {
				int idx_input = c*_WH_in*_WH_in + (h+r)*_WH_in + (w+s);
				int idx_weight = k*_C*_RS*_RS + c*_RS*_RS + r*_RS + s;
				int idx_input_1 = idx_input + _WH_in*_WH_in;
				int idx_weight_1 = idx_weight + _RS*_RS;
				int idx_input_2 = idx_input + 2*_WH_in*_WH_in;
				int idx_weight_2 = idx_weight + 2*_RS*_RS;
				int idx_input_3 = idx_input + 3*_WH_in*_WH_in;
				int idx_weight_3 = idx_weight + 3*_RS*_RS;
				output[idx_output] += input[idx_input] * weight[idx_weight];
				output[idx_output] += input[idx_input_1] * weight[idx_weight_1];
				output[idx_output] += input[idx_input_2] * weight[idx_weight_2];
				output[idx_output] += input[idx_input_3] * weight[idx_weight_3];
			}
		}
	}*/

	// Make access pattern simple
	/*output[idx_output] = 0;
	for (int r=0;r<_RS;r++) {
		for (int s=0;s<_RS;s++) {
			int idx_input = (h+r)*_WH_in + (w+s);
			int idx_weight = k*_C*_RS*_RS + r*_RS + s;
			for (int c=0;c<_C;c++) {
				output[idx_output] += input[idx_input] * weight[idx_weight];
				idx_input += _WH_in*_WH_in;
				idx_weight += _RS*_RS;
			}
		}
	}*/

	// Reduce memory access
	float tmp = 0.0;
	for (int r=0;r<_RS;r++) {
		for (int s=0;s<_RS;s++) {
			int idx_input = (h+r)*_WH_in + (w+s);
			int idx_weight = k*_C*_RS*_RS + r*_RS + s;
			for (int c=0;c<_C;c++) {
				tmp += input[idx_input] * weight[idx_weight];
				idx_input += _WH_in*_WH_in;
				idx_weight += _RS*_RS;
			}
		}
	}
	output[idx_output] = tmp;
}

void getConvolutionGold(const float* input,
						const float* weight,
						//const float* bias,
						const int _WH_in,
						const int _WH,
						const int _RS,
						const int _C,
						const int _K,
						float* output_gold)
{
	for (int k=0;k<_K;k++) {
		for (int h=0;h<_WH;h++) {
			for (int w=0;w<_WH;w++) {
				int idx_output = k*_WH*_WH + h*_WH + w;
				//output_gold[idx_output] = bias[k];
				output_gold[idx_output] = 0;
				for (int r=0;r<_RS;r++) {
					for (int s=0;s<_RS;s++) {
						for (int c=0;c<_C;c++) {
							int idx_input = c*_WH_in*_WH_in + (h+r)*_WH_in + (w+s);
							int idx_weight = k*_C*_RS*_RS + c*_RS*_RS + r*_RS + s;
							output_gold[idx_output] += input[idx_input] * weight[idx_weight];
						}
					}
				}
			}
		}
	}
}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int Conv() {
  // Allocate host memory for matrices A and B
  unsigned int size_A = WH_in*WH_in*C;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  //checkCudaErrors(cudaMallocHost(&h_A, mem_size_A));
  cudaMallocHost(&h_A, mem_size_A);
  unsigned int size_B = K*C*RS*RS;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  //checkCudaErrors(cudaMallocHost(&h_B, mem_size_B));
  cudaMallocHost(&h_B, mem_size_B);
  cudaStream_t stream;

  // Initialize host memory
  const float valB = 0.01f;
  //ConstantInit(h_A, size_A, 1.0f);
  for (int i=0;i<size_A;i++) h_A[i] = 1.0f;
  //ConstantInit(h_B, size_B, valB);
  for (int i=0;i<size_B;i++) h_B[i] = valB;

  // Allocate device memory
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C
  unsigned int mem_size_C = WH * WH * K * sizeof(float);
  float *h_C;
  //checkCudaErrors(cudaMallocHost(&h_C, mem_size_C));
  cudaMallocHost(&h_C, mem_size_C);

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A);
  cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B);
  cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C);
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // copy host memory to device

      cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream);

      cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream);

  // Setup execution parameters
  int block_size = 16;
  int block_size_c = 4;
  //dim3 threads(block_size_c, block_size, block_size);
  //dim3 grid(K / threads.x, WH / threads.y, WH / threads.z);
  dim3 threads(block_size, block_size, block_size_c);
  dim3 grid(WH / threads.x, WH / threads.y, K / threads.z);

  // Create and start timer
  printf("Computing result using CUDA Kernel...\n");

  // Performs warmup operation using matrixMul CUDA kernel
  if (block_size == 16) {
	  ConvCUDA
        <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, WH_in, WH, RS, C, K);
  } else {
	  ConvCUDA
    	<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, WH_in, WH, RS, C, K);
  }

  printf("done\n");
  cudaStreamSynchronize(stream);

  // Record the start event
  cudaEventRecord(start, stream);

  // Execute the kernel
  int nIter = 300;

  for (int j = 0; j < nIter; j++) {
    if (block_size == 16) {
    	ConvCUDA
      	  <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, WH_in, WH, RS, C, K);
    } else {
    	ConvCUDA
      	  <<<grid, threads, 0, stream>>>(d_C, d_A, d_B, WH_in, WH, RS, C, K);
    }
  }

  // Record the stop event
  cudaEventRecord(stop, stream);

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  float msecTotal = 0.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);


  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  printf("Conv Kernel Time= %.3f msec, WorkgroupSize= %u threads/block\n", msecPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host

      cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  printf("Checking computed result for correctness: ");
  float* gold_C = (float*)malloc(mem_size_C);
  getConvolutionGold(h_A, h_B, WH_in, WH, RS, C, K, gold_C);
  bool correct = true;

  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6;  // machine zero

  for (int i = 0; i < static_cast<int>(WH*WH*K); i++) {
    double abs_err = fabs(h_C[i] - gold_C[i]);
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_C[i], gold_C[i], eps);
      correct = false;
    }
  }

  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

  // Clean up memory
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf(
      "\nNOTE: The CUDA Samples are not meant for performance"
      "measurements. Results may vary when GPU Boost is enabled.\n");

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

/**
 * Program main
 */
int main(int argc, char **argv) {

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  //int dev = findCudaDevice(argc, (const char **)argv);
	int dev = 0;
  //int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB);
  int matrix_result = Conv();

  exit(matrix_result);
}
