/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Modifications and refactoring by "dialynnn".
// CUTLASS version: 2.5.0 (hard requirement)
// Architecture used: Volta (V100, sm_70)

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>
#include <random>
#include <iostream>
#include "cuda_utils.h"


// Convenience functions
using half_t = cutlass::half_t;

void convert_f32_to_f16(const float* in, half_t* out, int n){
    for (int i = 0; i < n; ++i) {
        out[i] = half_t(in[i]); 
    }
}


// ============================================================= 


// Constructing the GEMM
// Because of template hell, and CUTLASS hating it if we were to construct it inside a function, 
// we need to declare all of the stuff first outside of function first.
// Copying straight from NVIDIA's own sample w/ simplified comments.
using ElementAccumulator = float;       // <- data type of accumulator
using ElementComputeEpilogue = float;   // <- data type of epilogue operations
using ElementInputA = half_t;           // <- data type of elements in input matrix A
using ElementInputB = half_t;           // <- data type of elements in input matrix B
using ElementOutput = float;            // <- data type of elements in output matrix D

// Matrix layout
using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// Use tensor cores or SIMT cores
using MMAOp = cutlass::arch::OpClassTensorOp;

// CUDA SM architecture (We're on Volta)
using SmArch = cutlass::arch::Sm70;

// Threadblock tile size, warp tile size, and size of MMA op
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;           // <- warp tile M = 64, N = 64, K = 32 
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;                // <- MMA Op tile M = 8, N = 8, K = 4

// Threadblock swizzle (scheduler)
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// Epilogue
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                      // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,   // <- this is the number of elements per
                                                        // vectorized memory access. For half
                                                        // precision, it's 8 elements. This becomes
                                                        // the vector width of math instructions in
                                                        // epilogue too
    ElementAccumulator,                                 // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Num of stages/pipelines
constexpr int NumStages = 2;

// Declaring the entire GEMM
using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                        LayoutInputA,
                                        ElementInputB,
                                        LayoutInputB,
                                        ElementOutput,
                                        LayoutOutput,
                                        ElementAccumulator,
                                        MMAOp,
                                        SmArch,
                                        ShapeMMAThreadBlock,
                                        ShapeMMAWarp,
                                        ShapeMMAOp,
                                        EpilogueOp,
                                        SwizzleThreadBlock,
                                        NumStages>;


// ============================================================= 


// This is possible, by the way. Great for compartmentalizations!
static void cutlass_gemm_baseline(
    __restrict__ half_t* dA, __restrict__ half_t* dB,
    float* dC, float* dD,
    const int M, const int N, const int K){

    // GEMM problem size
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // Initialize alpha and beta for dot product computation
    // No need to initalize it on main
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;

    // GEMM arguments tuple
    // For reference: D = alpha * A * B + beta * C
    typename Gemm::Arguments arguments{problem_size,
                                        dA,  
                                        dB,  
                                        dC,  
                                        dD,  
                                        {alpha, beta},          
                                        split_k_slices};        

    // Workspace memory
    // This would be different on higher versions (ie., v3.0.0)
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    uint8_t* workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));

    // Instantiate CUTLASS kernel depending on templates
    Gemm gemm_op;

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    // Initialize CUTLASS kernel with arguments and workspace pointer
    status = gemm_op.initialize(arguments, workspace);
    CUTLASS_CHECK(status);

    // Launch initialized CUTLASS kernel
    status = gemm_op();
    CUTLASS_CHECK(status);

    // Clear workspace
    cudaFree(workspace);
}


int main(){
    // Dims
    const int M = 4096;
    const int N = 128;
    const int K = 4096;
    
    // Allocate GPU pointers
    half_t* dA = nullptr;
    half_t* dB = nullptr;
    float* dC = nullptr;  // Can be kept empty, usually omitted
    float* dD = nullptr;  // Actual output

    // For the input values (host)
    std::vector<float> hA(M * K);
    std::vector<float> hB(K * N);

    // GPU values
    std::vector<half_t> gpu_hA(M * K);
    std::vector<half_t> gpu_hB(K * N);
    std::vector<float> gpu_hC(M * N);  // Usually omitted
    std::vector<float> gpu_hD(M * N);

    // Initialize A and B with deterministic values
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float val = (i + k) * 0.001f;       
            hA[i * K + k] = val; 
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            float val = (k - j) * 0.002f;
            hB[k * N + j] = val;
        }
    }

    // Not possible to run half_t directly on the loop, so we put it in a function
    convert_f32_to_f16(hA.data(), gpu_hA.data(), {M * K});
    convert_f32_to_f16(hB.data(), gpu_hB.data(), {K * N});

    // Allocate memory
    CUDA_CHECK(cudaMalloc(&dA, gpu_hA.size() * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dB, gpu_hB.size() * sizeof(half_t)));
    CUDA_CHECK(cudaMalloc(&dC, gpu_hC.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dD, gpu_hD.size() * sizeof(float)));  // Output

    CUDA_CHECK(cudaMemcpy(dA, gpu_hA.data(), gpu_hA.size() * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, gpu_hB.data(), gpu_hB.size() * sizeof(half_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, gpu_hC.size() * sizeof(float)));  // Intermediate, can be kept zero
    CUDA_CHECK(cudaMemset(dD, 0, gpu_hD.size() * sizeof(float)));  // Output

    // Cutlass matmul + jank measurement method
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    cutlass_gemm_baseline(dA, dB, dC, dD, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    // We get the entire time after the devices are sync-ed
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Measure runtime
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); 
    printf("Elapsed time: %f ms\n", ms);
    printf("Throughput: %f TFLOP/s\n", ((float) 2 * M * N * K)/(ms*1e9));

    // Clean up
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dD));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop)); 

    return 0;
}

