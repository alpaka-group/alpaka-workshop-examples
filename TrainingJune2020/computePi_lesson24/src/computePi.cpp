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

    // Memory allocation on host to be done here,
    // will be added in lesson 25

    // Initialization of point data on host to be done here,
    // will be added in lesson 25

    // Memory allocation on device to be done here,
    // will be added in lesson 25

    // Memory copy from host to device to be done here,
    // will be added in lesson 25

    // Kernel to be executed here, will be added in lesson 26

    // Memory copy from device to host to be done here,
    // will be added in lesson 25

    // Wait until all operations in the queue are finished.
    // This call is redundant for a blocking queue
    // Here use alpaka:: because of an issue on macOS
    alpaka::wait::wait(queue);

    // Results to be integrated on host
    // and printed here, will be added in lesson 26

    return 0;
}
