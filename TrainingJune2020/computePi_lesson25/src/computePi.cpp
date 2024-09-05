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

// Alpaka kernel defines operations to be executed concurrently on a device
// It is a C++ functor: the entry point is operator()
struct PixelFinderKernel {
    // Alpaka accelerator is the required first parameter for all kernels.
    // It is provided by alpaka automatically.
    // For portability its type is a template parameter.
    // ALPAKA_FN_ACC prefix is required for all functions that run on device.
    // Kernels are required to be const and return void
    template<typename Acc>
    ALPAKA_FN_ACC void operator()(Acc const & acc,
        Points points, float r) const {
        // This function body will be executed by all threads concurrently
        using namespace alpaka;

        // Thread index in the grid (among all threads)
        uint32_t gridThreadIdx = idx::getIdx<Grid, Threads>(acc)[0];

        // Read inputs for the current threads to work on
        // For simplicity we assume the total number of threads
        // is equal to the number of points
        float x = points.x[gridThreadIdx];
        float y = points.y[gridThreadIdx];

        // Note acc parameter to sqrt, same for other math functions
        float d = math::sqrt(acc, x * x + y * y);

        // Compute and write output
        bool isInside = (d <= r);
        points.inside[gridThreadIdx] = isInside;
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

    // This is to disable unused warnings
    alpaka::ignore_unused(pointsAcc);

    // Copy x, y buffers from host to device
    mem::view::copy(queue, xBufferAcc, xBufferHost, bufferExtent);
    mem::view::copy(queue, yBufferAcc, yBufferHost, bufferExtent);

    // Kernel to be executed here, will be added in lesson 26

    // Copy inside buffer from device to host
    mem::view::copy(queue, insideBufferHost, insideBufferAcc, bufferExtent);

    // Wait until all operations in the queue are finished.
    // This call is redundant for a blocking queue
    // Here use alpaka:: because of an issue on macOS
    alpaka::wait::wait(queue);

    // Results to be integrated on host
    // and printed here, will be added in lesson 26

    return 0;
}
