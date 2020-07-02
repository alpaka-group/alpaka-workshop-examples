/* Copyright 2019-2020 Benjamin Worpitz, Erik Zenker, Sergei Bastrakov
 *
 * This file exemplifies usage of Alpaka.
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED “AS IS” AND ISC DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL ISC BE LIABLE FOR ANY
 * SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
 * WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
 * ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR
 * IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
 */

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <iostream>

// Alpaka kernel defines operations to be executed concurrently on a device
// It is a C++ functor: the entry point is operator()
struct HelloWorldKernel {
    // Alpaka accelerator is the required first parameter for all kernels.
    // It is provided by alpaka automatically.
    // For portability its type is a template parameter.
    // ALPAKA_FN_ACC prefix is required for all functions that run on device.
    // Kernels are required to be const and return void
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const & acc) const {
        // This function body will be executed by all threads concurrently
        using namespace alpaka;
        // The acc parameter is used to access alpaka abstractions in kernels,
        // in this case thread indexing

        // The following changes to switch to 2D indexing are done in lesson 22

        uint32_t gridThreadIdxY = idx::getIdx<Grid, Threads>(acc)[0];
        uint32_t gridThreadIdxX = idx::getIdx<Grid, Threads>(acc)[1];
        printf("Hello, World from alpaka thread (%u, %u)!\n", gridThreadIdxY, gridThreadIdxX);

        // It is possible to get index type from the accelerator,
        // but for simplicity we just repeat the type here
        using Vec = vec::Vec<dim::DimInt<2>, uint32_t>;
        Vec gridBlockIdx = idx::getIdx<Grid, Blocks>(acc);
        Vec blockThreadIdx = idx::getIdx<Block, Threads>(acc);
        printf("Hello, World from alpaka thread (%u, %u) in block (%u, %u)!\n",
            blockThreadIdx[0], blockThreadIdx[1], gridBlockIdx[0], gridBlockIdx[1]);
    }
};

int main() {
    // For code brevity, all alpaka API is in namespace alpaka
    using namespace alpaka;

    // Define dimensionality and type of indices to be used in kernels
    // Change to 2d indexing in lesson 22
    using Dim = dim::DimInt<2>;
    using Idx = uint32_t;

    // Define alpaka accelerator type, which corresponds to the underlying programming model
    using Acc = acc::AccCpuOmp2Blocks<Dim, Idx>;
    // Other options instead of AccCpuOmp2Blocks are
    // - AccGpuCudaRt
    // - AccCpuThreads
    // - AccCpuFibers
    // - AccCpuOmp2Threads
    // - AccCpuOmp4
    // - AccCpuTbbBlocks
    // - AccCpuSerial

    // Select the first device available on a system, for the chosen accelerator
    auto const device = pltf::getDevByIdx<Acc>(0u);

    // Define type for a queue with requested properties:
    // in this example we require the queue to be blocking the host side
    // while operations on the device (kernels, memory transfers) are running
    using Queue = queue::Queue<Acc, queue::Blocking>;
    // Create a queue for the device
    auto queue = Queue{device};

    // Define kernel execution configuration of blocks,
    // threads per block, and elements per thread
    // The kernel launch configuration is 2d in lesson 22
    // Note that the constructor arguments have to be of type Idx,
    // not matching types may cause template substitution errors
    auto blocksPerGrid = vec::Vec<Dim, Idx>{2u, 4u};
    auto threadsPerBlock = vec::Vec<Dim, Idx>{1u, 1u};
    auto elementsPerThread = vec::Vec<Dim, Idx>{1u, 1u};
    using WorkDiv = workdiv::WorkDivMembers<Dim, Idx>;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};

    // Instantiate the kernel object
    HelloWorldKernel helloWorldKernel;
    // Create a task to run the kernel with the given work division;
    // creating a task does not put it for execution
    auto taskRunKernel = kernel::createTaskKernel<Acc>(workDiv, helloWorldKernel);

    // Enqueue the kernel execution task.
    // The kernel's operator() will be run concurrently
    // on the device associated with the queue.
    queue::enqueue(queue, taskRunKernel);

    // Wait until all operations in the queue are finished.
    // This call is redundant for a blocking queue
    // Here use alpaka:: because of an issue on macOS
    alpaka::wait::wait(queue);

    return 0;
}
