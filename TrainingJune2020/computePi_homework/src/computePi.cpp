/* Copyright 2019-2020 Benjamin Worpitz, Erik Zenker, Jan Stephan,
 *                     Sergei Bastrakov
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
#include <random>

// Structure with memory buffers for inputs (x, y) and
// outputs (inside) of the kernel
struct Points {
    float * x;
    float * y;
    bool * inside;
};

// Since this homework aims to illustrate general workload distribution patterns,
// we move processing of a single point to a separate function for better demonstration.
template<typename Acc>
ALPAKA_FN_ACC void processPoint(Acc const & acc, Points points, float r, uint32_t idx)
{
    using namespace alpaka;
    float x = points.x[idx];
    float y = points.y[idx];
    float d = math::sqrt(acc, x * x + y * y);
    bool isInside = (d <= r);
    points.inside[idx] = isInside;
}

// Kernel as used in lesson 26, one thread processes one point
// Implements a simplified case with number of points being equal to number of threads.
// This kernel is not suitable for the general case,
// as the number of points has to be a multiple of the block size
struct PixelFinderKernelOnePointPerThreadSimplified {
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const & acc, Points points, float r) const
    {
        using namespace alpaka;
        // Thread index in the grid (among all threads)
        uint32_t gridThreadIdx = idx::getIdx<Grid, Threads>(acc)[0];

        // Each thread processes a single point with the corresponding index
        processPoint(acc, points, r, gridThreadIdx);
    }
};

// This is a general version of PixelFinderKernelOnePointPerThreadSimplified:
// one thread processes one point, number of threads is equal or larger than the number of points.
// Now we need to take the number of points n as input.
struct PixelFinderKernelOnePointPerThread {
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const & acc, Points points, float r, uint32_t n) const
    {
        using namespace alpaka;
        // Thread index in the grid (among all threads)
        uint32_t gridThreadIdx = idx::getIdx<Grid, Threads>(acc)[0];

        // In the general case we need to check if the current thread has work to do
        if (gridThreadIdx < n)
            processPoint(acc, points, r, gridThreadIdx);
    }
};

// This is a general version of the kernel that works for any work division.
// It employs a widely used approach to workload distribution in alpaka (and CUDA) kernels
// Note that this kernel does not employ the alpaka element layer yet
struct PixelFinderKernelMultiplePointsPerThread {
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const & acc, Points points, float r, uint32_t n) const
    {
        using namespace alpaka;
        // Thread index in the grid (among all threads)
        uint32_t gridThreadIdx = idx::getIdx<Grid, Threads>(acc)[0];
        uint32_t gridThreadExtent = workdiv::getWorkDiv<Grid, Threads>(acc)[0];

        // Strided loop over points, a widely used technique
        for (uint32_t idx = gridThreadIdx; idx < n; idx += gridThreadExtent)
            processPoint(acc, points, r, idx);
    }
};

// This is a general version of the PixelFinderKernelMultiplePointsPerThread kernel,
// which also employs alpaka element layer.
// It employs striding and loop blocking, to allow efficient processing
// on both CPUs and GPU with a proper choice of element extent
struct PixelFinderKernelMultiplePointsPerThreadElements {
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const & acc, Points points, float r, uint32_t n) const
    {
        using namespace alpaka;
        // Thread index in the grid (among all threads)
        uint32_t gridThreadIdx = idx::getIdx<Grid, Threads>(acc)[0];
        uint32_t gridThreadExtent = workdiv::getWorkDiv<Grid, Threads>(acc)[0];
        uint32_t threadElementExtent = workdiv::getWorkDiv<Thread, Elems>(acc)[0];

        // Strided loop over points
        for (uint32_t idx = gridThreadIdx * threadElementExtent; idx < n;
            idx += gridThreadExtent * threadElementExtent)
        {
            // Loop blocking to process a contiguous chunk after each "jump" in the outer loop
            for (uint32_t i = idx; (i < idx + threadElementExtent) && (i < n); i++)
                processPoint(acc, points, r, i);
        }
    }
};


int main() {
    // For code brevity, all alpaka API is in namespace alpaka
    using namespace alpaka;

    // Define dimensionality and type of indices to be used in kernels
    using Dim = dim::DimInt<1>;
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

    // Number of points
    uint32_t n = 10000;

    // Circle radius
    float r = 10.0f;

    // Create a device for host for memory allocation, using the first CPU available
    auto devHost = pltf::getDevByIdx<dev::DevCpu>(0u);

    // Allocate memory on the host side:
    // the first template parameter is data type of buffer elements,
    // the second is internal indexing type
    vec::Vec<Dim, Idx> bufferExtent{n};
    auto xBufferHost = mem::buf::alloc<float, Idx>(devHost, bufferExtent);
    auto yBufferHost = mem::buf::alloc<float, Idx>(devHost, bufferExtent);
    auto insideBufferHost = mem::buf::alloc<bool, Idx>(devHost, bufferExtent);

    // Get raw pointers to memory buffers on host and put into a structure
    Points pointsHost;
    pointsHost.x = mem::view::getPtrNative(xBufferHost);
    pointsHost.y = mem::view::getPtrNative(yBufferHost);
    pointsHost.inside = mem::view::getPtrNative(insideBufferHost);

    // Generate input x, y randomly in [0, r]
    std::random_device rd;
    std::mt19937 generator{rd()};
    std::uniform_real_distribution<float> distribution(0.0f, r);
    for (auto idx = 0u; idx < n; idx++)
    {
        pointsHost.x[idx] = distribution(generator);
        pointsHost.y[idx] = distribution(generator);
    }

    // Allocate memory on the device side,
    // note symmetry to host
    auto xBufferAcc = mem::buf::alloc<float, Idx>(device, bufferExtent);
    auto yBufferAcc = mem::buf::alloc<float, Idx>(device, bufferExtent);
    auto insideBufferAcc = mem::buf::alloc<bool, Idx>(device, bufferExtent);

    // Get raw pointers to memory buffers device host and put into a structure,
    // note symmetry to host
    Points pointsAcc;
    pointsAcc.x = mem::view::getPtrNative(xBufferAcc);
    pointsAcc.y = mem::view::getPtrNative(yBufferAcc);
    pointsAcc.inside = mem::view::getPtrNative(insideBufferAcc);

    // Start time measurement
    auto start = std::chrono::steady_clock::now();

    // Copy x, y buffers from host to device
    mem::view::copy(queue, xBufferAcc, xBufferHost, bufferExtent);
    mem::view::copy(queue, yBufferAcc, yBufferHost, bufferExtent);

    // Define kernel execution configuration of blocks,
    // threads per block, and elements per thread
    // Note that different kernels pose different requirements to the workDiv
    uint32_t blocksPerGrid = n;
    uint32_t threadsPerBlock = 1;
    uint32_t elementsPerThread = 1;
    using WorkDiv = workdiv::WorkDivMembers<Dim, Idx>;
    auto workDiv = WorkDiv{blocksPerGrid, threadsPerBlock, elementsPerThread};

    // Instantiate the kernel object
    PixelFinderKernelOnePointPerThreadSimplified pixelFinderKernel;
    // Create a task to run the kernel with the given work division;
    // creating a task does not put it for execution
    // Note that all kernels but PixelFinderKernelOnePointPerThreadSimplified
    // additionally take n as the last argument
    auto taskRunKernel = kernel::createTaskKernel<Acc>(workDiv, pixelFinderKernel, pointsAcc, r/*,  n*/);

    // Enqueue the kernel execution task.
    // The kernel's operator() will be run concurrently
    // on the device associated with the queue.
    queue::enqueue(queue, taskRunKernel);

    // Copy inside buffer from device to host
    mem::view::copy(queue, insideBufferHost, insideBufferAcc, bufferExtent);

    // Wait until all operations in the queue are finished.
    // This call is redundant for a blocking queue
    // Here use alpaka:: because of an issue on macOS
    alpaka::wait::wait(queue);

    // Compute Pi on host
    uint32_t P = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (pointsHost.inside[i])
            ++P;
    }
    float pi = 4.f * P / n;

    // Finish time measurements
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Output results
    std::cout << "Computed pi is " << pi << "\n";
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    return 0;
}
