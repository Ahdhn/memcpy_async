#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <numeric>
#include "gtest/gtest.h"
#include "helper.h"


__device__ void compute(uint32_t*       global_out,
                        uint32_t const* shared_in,
                        const uint32_t  local_idx,
                        const uint32_t  block_size)
{
    // Computes using all values of current batch from shared memory.
    // Stores this thread's result back to global memory.
    uint32_t out = shared_in[local_idx] + shared_in[block_size - local_idx - 1];
    global_out[local_idx] = out;
}

__global__ void without_memcpy_async(uint32_t*       global_out,
                                     uint32_t const* global_in,
                                     uint32_t        size,
                                     uint32_t        num_batch_per_block)
{
    auto grid  = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();

    // input size fits num_batch_per_block * grid_size
    assert(size == num_batch_per_block * grid.size());

    // block.size() * sizeof(int) bytes
    extern __shared__ uint32_t shared[];

    uint32_t local_idx = block.thread_rank();

    for (uint32_t batch = 0; batch < num_batch_per_block; ++batch) {
        // Compute the index of the current batch for this block in global
        // memory:
        uint32_t block_batch_idx =
            block.group_index().x * block.size() + grid.size() * batch;

        uint32_t global_idx = block_batch_idx + local_idx;

        shared[local_idx] = global_in[global_idx];

        // Wait for all copies to complete
        block.sync();

        // Compute and write result to global memory
        // compute(global_out + block_batch_idx, shared, local_idx);
        global_out[global_idx] =
            shared[local_idx] + shared[block.size() - local_idx - 1];

        // Wait for compute using shared memory to finish
        block.sync();
    }
}


__global__ void testKernel()
{
}

TEST(Test, memcpy_async)
{
    uint32_t size                = 1024 * 1024;
    uint32_t block_size          = 256;
    uint32_t num_batch_per_block = 2;

    uint32_t *d_in(nullptr), *d_out(nullptr);
    CUDA_ERROR(cudaMalloc((void**)&d_in, size * sizeof(uint32_t)));
    CUDA_ERROR(cudaMalloc((void**)&d_out, size * sizeof(uint32_t)));

    std::vector<uint32_t> batch(block_size);
    std::iota(std::begin(batch), std::end(batch), 0);
    std::vector<uint32_t> h_in(size, 0), h_out(size, 0);
    uint32_t              num_segments = size / block_size;
    for (uint32_t i = 0; i < num_segments; ++i) {
        std::copy(batch.begin(), batch.end(), h_in.data() + i * batch.size());
    }
    uint32_t smem_bytes = block_size * sizeof(uint32_t);
    uint32_t grid_size  = DIVIDE_UP(size, block_size * num_batch_per_block);

    CUDA_ERROR(cudaMemcpy(
        d_in, h_in.data(), size * sizeof(uint32_t), cudaMemcpyHostToDevice));

    for (uint32_t method = 0; method < 2; ++method) {

        CUDA_ERROR(cudaMemset(d_out, 0, size * sizeof(uint32_t)));

        CUDATimer timer;

        timer.start();
        if (method == 0) {
            without_memcpy_async<<<grid_size, block_size, smem_bytes>>>(
                d_out, d_in, size, num_batch_per_block);
        } else if (method == 1) {

        }
        timer.stop();
        CUDA_ERROR(cudaMemcpy(h_out.data(),
                              d_out,
                              size * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < h_out.size(); ++i) {
            ASSERT_EQ(h_out[i], block_size - 1);
        }
    }

    auto err = cudaDeviceSynchronize();
    EXPECT_EQ(err, cudaSuccess);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
