#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

class GpuMemoryPool {
  public:
    GpuMemoryPool() = default;

    ~GpuMemoryPool();

    GpuMemoryPool(GpuMemoryPool const &) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool const &) = delete;
    GpuMemoryPool(GpuMemoryPool &&) = delete;
    GpuMemoryPool &operator=(GpuMemoryPool &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    std::vector<void *> allocations_;
    std::vector<size_t> capacities_;
    size_t next_idx_ = 0;
};

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void render_cpu(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    float const *circle_red,
    float const *circle_green,
    float const *circle_blue,
    float const *circle_alpha,
    float *img_red,
    float *img_green,
    float *img_blue) {

    // Initialize background to white
    for (int32_t pixel_idx = 0; pixel_idx < width * height; pixel_idx++) {
        img_red[pixel_idx] = 1.0f;
        img_green[pixel_idx] = 1.0f;
        img_blue[pixel_idx] = 1.0f;
    }

    // Render circles
    for (int32_t i = 0; i < n_circle; i++) {
        float c_x = circle_x[i];
        float c_y = circle_y[i];
        float c_radius = circle_radius[i];
        for (int32_t y = int32_t(c_y - c_radius); y <= int32_t(c_y + c_radius + 1.0f);
             y++) {
            for (int32_t x = int32_t(c_x - c_radius); x <= int32_t(c_x + c_radius + 1.0f);
                 x++) {
                float dx = x - c_x;
                float dy = y - c_y;
                if (!(0 <= x && x < width && 0 <= y && y < height &&
                      dx * dx + dy * dy < c_radius * c_radius)) {
                    continue;
                }
                int32_t pixel_idx = y * width + x;
                float pixel_red = img_red[pixel_idx];
                float pixel_green = img_green[pixel_idx];
                float pixel_blue = img_blue[pixel_idx];
                float pixel_alpha = circle_alpha[i];
                pixel_red =
                    circle_red[i] * pixel_alpha + pixel_red * (1.0f - pixel_alpha);
                pixel_green =
                    circle_green[i] * pixel_alpha + pixel_green * (1.0f - pixel_alpha);
                pixel_blue =
                    circle_blue[i] * pixel_alpha + pixel_blue * (1.0f - pixel_alpha);
                img_red[pixel_idx] = pixel_red;
                img_green[pixel_idx] = pixel_green;
                img_blue[pixel_idx] = pixel_blue;
            }
        }
    }
}

/// <--- your code here --->

// PSeudo Code

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace circles_gpu {

#define THREADS_SCAN (4 * 32)
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define PAD 32
#define SHMEM_PADDING(idx) ((idx) + ((idx) / PAD))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* TODO: your GPU kernels here... */

__global__ void
fill_data(const uint8_t *__restrict__ in, uint32_t *__restrict__ out, size_t n) {
    size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;

    // Fast path if base pointer is 4B-aligned
    uintptr_t base = reinterpret_cast<uintptr_t>(in);
    bool aligned = (base & 3u) == 0;

    if (i + 3 < n) {

        const uint8_t *p = in + i;
        out[i + 0] = uint32_t(p[0]);
        out[i + 1] = uint32_t(p[1]);
        out[i + 2] = uint32_t(p[2]);
        out[i + 3] = uint32_t(p[3]);
        // }
    } else {
        // Tail
        for (int k = 0; k < 4 && i + k < n; ++k)
            out[i + k] = uint32_t(in[i + k]);
    }
}

#define THREADS 128
__global__ void scan_block_write_totals(
    size_t n,
    const uint32_t *__restrict__ x,
    uint32_t *__restrict__ out,
    uint32_t *__restrict__ block_sums) {

    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int WARP_SIZE = 32;

    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t v = (idx < n) ? x[idx] : 0u;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int warp = threadIdx.x >> 5; // warp id within block

// Intra-warp inclusive scan via shuffles
#pragma unroll
    for (int d = 1; d < WARP_SIZE; d <<= 1) {
        uint32_t u = __shfl_up_sync(FULL_MASK, v, d);
        if (lane >= d)
            v += u;
    }

    // Accumulate warp sums in shared
    __shared__ uint32_t warp_sums[THREADS / WARP_SIZE];
    if (lane == WARP_SIZE - 1)
        warp_sums[warp] = v;
    __syncthreads();

    // Warp 0 scans warp_sums
    uint32_t warp_off = 0;
    if (warp == 0) {
        const int WARPS = THREADS / WARP_SIZE;
        uint32_t t = (lane < WARPS) ? warp_sums[lane] : 0u;
#pragma unroll
        for (int d = 1; d < WARP_SIZE; d <<= 1) {
            uint32_t u = __shfl_up_sync(FULL_MASK, t, d);
            if (lane >= d)
                t += u;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();

    if (warp > 0)
        warp_off = warp_sums[warp - 1]; // what to sum
    v += warp_off;

    if (idx < n)
        out[idx] = v;
    // scan per block

    // Last active thread in the block writes the block's total
    if (threadIdx.x == blockDim.x - 1 || idx == n - 1) {
        block_sums[blockIdx.x] = v;
    }
}

__global__ void
scan_single(size_t m, const uint32_t *__restrict__ in, uint32_t *__restrict__ out) {
    extern __shared__ uint32_t smem[]; // holds warp sums (WARPS elements)
    const unsigned FULL_MASK = 0xFFFFFFFFu;
    const int WARP_SIZE = 32;

    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid >> 5;
    const int WARPS = (blockDim.x + 31) / 32;

    uint32_t v = (size_t(tid) < m) ? in[tid] : 0u;

// Intra-warp inclusive
#pragma unroll
    for (int d = 1; d < WARP_SIZE; d <<= 1) {
        uint32_t u = __shfl_up_sync(FULL_MASK, v, d);
        if (lane >= d)
            v += u;
    }

    if (lane == WARP_SIZE - 1)
        smem[warp] = v;
    __syncthreads();

    uint32_t warp_off = 0;
    if (warp == 0) {
        uint32_t t = (lane < WARPS) ? smem[lane] : 0u;
#pragma unroll
        for (int d = 1; d < WARP_SIZE; d <<= 1) {
            uint32_t u = __shfl_up_sync(FULL_MASK, t, d);
            if (lane >= d)
                t += u;
        }
        smem[lane] = t;
    }
    __syncthreads();

    if (warp > 0)
        warp_off = smem[warp - 1];
    v += warp_off;

    if (size_t(tid) < m)
        out[tid] = v;
}
__global__ void uniform_add(
    size_t n,
    uint32_t *__restrict__ out,
    const uint32_t *__restrict__ block_prefix) {
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    const uint32_t add = (blockIdx.x == 0) ? 0u : block_prefix[blockIdx.x - 1];
    out[idx] += add;
}

inline size_t get_workspace_size_scan(size_t n, int threads) {
    const size_t blocksA = CEIL_DIV(n, (size_t)threads);
    const size_t blocksB0 = CEIL_DIV(blocksA, (size_t)threads);
    const size_t u32_elems = n + blocksA + blocksA + blocksB0 + blocksB0;
    return u32_elems * sizeof(uint32_t);
}

uint32_t *launch_scan(size_t n, const uint8_t *d_in, void *workspace, int threads = 128) {

    auto *base = reinterpret_cast<uint32_t *>(workspace);
    const size_t blocksA = CEIL_DIV(n, (size_t)threads);
    const size_t blocksB0 = CEIL_DIV(blocksA, (size_t)threads);

    uint32_t *d_out_u32 = base;
    uint32_t *d_block_sums = d_out_u32 + n;
    uint32_t *d_prefix_L0 = d_block_sums + blocksA;
    uint32_t *d_tmp0 = d_prefix_L0 + blocksA;
    uint32_t *d_tmp1 = d_tmp0 + blocksB0;
    const int t = threads;
    int b = int(CEIL_DIV(n, (size_t)(4 * t)));

    // covert to uint32

    if (b > 0)
        fill_data<<<b, t>>>(d_in, d_out_u32, n);

    b = int(blocksA);
    if (b > 0)
        scan_block_write_totals<<<b, t>>>(n, d_out_u32, d_out_u32, d_block_sums);

    const int T = 1024;
    if (blocksA <= 1024) {

        const int WARPS = (T + 31) / 32;
        const int smem = WARPS * sizeof(uint32_t);
        scan_single<<<1, T, smem>>>(blocksA, d_block_sums, d_prefix_L0);
    } else {
        // Level 0

        const int t = threads;
        const int b = int(blocksB0);
        scan_block_write_totals<<<b, t>>>(blocksA, d_block_sums, d_prefix_L0, d_tmp0);

        // Level 1

        const int WARPS = (T + 31) / 32;
        const int smem = WARPS * sizeof(uint32_t);
        scan_single<<<1, T, smem>>>(blocksB0, d_tmp0, d_tmp1);

        // Propagate
        uniform_add<<<int(blocksB0), threads>>>(blocksA, d_prefix_L0, d_tmp1);
    }

    uniform_add<<<int(blocksA), threads>>>(n, d_out_u32, d_prefix_L0);

    return d_out_u32;
}

#define THREADS_X 32
#define THREADS_Y 32
#define TILE_SIZE 64

/* TODO: your GPU kernels here... */

__global__ void tile_coverage(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    uint8_t *circle_map,
    int32_t num_tiles) {

    int32_t circle = blockIdx.x * blockDim.x + threadIdx.x;

    size_t stride = (size_t)n_circle;
    size_t total = (size_t)num_tiles * stride;

    if (circle < n_circle) {

        int tile_width = width / TILE_SIZE;
        int tile_height = height / TILE_SIZE;

        float x = circle_x[circle];
        float y = circle_y[circle];
        float rad = circle_radius[circle];

        int left = (int)MAX((x - rad) / TILE_SIZE, 0);
        int right = (int)MIN((x + rad) / TILE_SIZE, tile_width - 1);
        int top = (int)MAX((y - rad) / TILE_SIZE, 0);
        int bot = (int)MIN((y + rad) / TILE_SIZE, tile_height - 1);

        // go through each corner of the bounding box
        for (int x = left; x < right + 1; x += 1) {
            for (int y = top; y < bot + 1; y += 1) {
                int tile = y * tile_width + x;

                size_t idx = (size_t)tile * stride + (size_t)circle;

                circle_map[idx] = (idx < total);
            }
        }
    }
}
#define MAX_CIRCLES_PER_TILE 44000
__global__ void compact_stream(
    uint32_t n_circle,
    uint32_t *scanned_idxs,
    uint32_t *compacted_idxs,
    uint32_t *num_circles_per_tile) {

    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_circle) {
        if (idx == 0) {
            compacted_idxs[0] = 0;
        } else if (scanned_idxs[idx] != scanned_idxs[idx - 1]) {
            uint32_t c_idx = scanned_idxs[idx];

            compacted_idxs[c_idx - 1] = idx;
        }
    }
    if (idx == n_circle - 1) {
        *num_circles_per_tile = scanned_idxs[idx];
    }
}

#define CIRCLE_STATS 8
__global__ void render_pixels(
    uint32_t *num_circles_per_tile,
    uint32_t n_circle,
    uint32_t num_tiles,
    int32_t width,
    int32_t height,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    float const *circle_red,
    float const *circle_green,
    float const *circle_blue,
    float const *circle_alpha,
    float *img_red,
    float *img_green,
    float *img_blue,
    uint32_t *compacted_stream

    // float **x_coal_ptr
) {

    int blockToTile =
        TILE_SIZE / THREADS_X; // assume threads x and threads y are the same

    int32_t tile_idx = blockIdx.y * gridDim.x + blockIdx.x;

    int32_t base_pixel_row = (blockIdx.y * blockDim.y) * blockToTile + threadIdx.y;
    int32_t base_pixel_col = (blockIdx.x * blockDim.x) * blockToTile + threadIdx.x;

    extern __shared__ __align__(16) char shmem_raw[];
    float *shmem = reinterpret_cast<float *>(shmem_raw);
    const int threads = blockDim.x * blockDim.y;

    uint32_t n_t_circles = num_circles_per_tile[tile_idx];
    uint32_t *tile_circles = compacted_stream + tile_idx * MAX_CIRCLES_PER_TILE;

    float red_out[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float green_out[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float blue_out[4] = {1.0f, 1.0f, 1.0f, 1.0f};

    for (int chunk = 0; chunk < n_t_circles; chunk += threads) {

        int tid = (threadIdx.y * blockDim.x + threadIdx.x);
        int circle_id_id = chunk + tid;

        if (circle_id_id < n_t_circles) {
            uint32_t c_idx = tile_circles[circle_id_id];

            float4 circle_xyra = make_float4(
                circle_x[c_idx],
                circle_y[c_idx],
                circle_radius[c_idx],
                circle_alpha[c_idx]);

            float4 circle_rgbp = make_float4(
                circle_red[c_idx],
                circle_green[c_idx],
                circle_blue[c_idx],
                0.0f // padding (unused)
            );

            reinterpret_cast<float4 *>(&shmem[CIRCLE_STATS * tid])[0] = circle_xyra;
            reinterpret_cast<float4 *>(&shmem[CIRCLE_STATS * tid])[1] = circle_rgbp;
        }

        int loop_iters = threads;
        if (chunk + threads >= n_t_circles) {
            loop_iters = n_t_circles - chunk; // the remainder
        }

        __syncthreads();
        for (int b_i = 0; b_i < blockToTile; ++b_i) {
            for (int b_j = 0; b_j < blockToTile; ++b_j) {

                int b_id = b_i * blockToTile + b_j;
                int32_t pixel_row = base_pixel_row + b_i * THREADS_Y;
                int32_t pixel_col = base_pixel_col + b_j * THREADS_X;

                for (int i = 0; i < loop_iters; ++i) {

                    // uint32_t c_idx = tile_circles[i];
                    float *circle_stats = shmem + CIRCLE_STATS * i;

                    float4 xyra = reinterpret_cast<const float4 *>(circle_stats)[0];
                    float4 rgbp = reinterpret_cast<const float4 *>(circle_stats)[1];

                    float x = xyra.x;
                    float y = xyra.y;
                    float rad = xyra.z;
                    float alpha = xyra.w;

                    float c_red = rgbp.x;
                    float c_green = rgbp.y;
                    float c_blue = rgbp.z;

                    float dy = pixel_row - y;
                    float dx = pixel_col - x;

                    if (!(dy * dy + dx * dx < rad * rad))
                        continue;

                    red_out[b_id] = c_red * alpha + red_out[b_id] * (1.0f - alpha);
                    green_out[b_id] = c_green * alpha + green_out[b_id] * (1.0f - alpha);
                    blue_out[b_id] = c_blue * alpha + blue_out[b_id] * (1.0f - alpha);
                }
            }
        }
        // __syncthreads();
    }

    for (int i = 0; i < blockToTile; ++i) {
        for (int j = 0; j < blockToTile; ++j) {

            int32_t pixel_row = base_pixel_row + i * THREADS_Y;
            int32_t pixel_col = base_pixel_col + j * THREADS_X;

            int32_t pixel_idx = pixel_row * width + pixel_col;
            int b_id = i * blockToTile + j;

            img_red[pixel_idx] = red_out[b_id];
            img_green[pixel_idx] = green_out[b_id];
            img_blue[pixel_idx] = blue_out[b_id];
        }
    }
} // namespace circles_gpu

void launch_render(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,      // pointer to GPU memory
    float const *circle_y,      // pointer to GPU memory
    float const *circle_radius, // pointer to GPU memory
    float const *circle_red,    // pointer to GPU memory
    float const *circle_green,  // pointer to GPU memory
    float const *circle_blue,   // pointer to GPU memory
    float const *circle_alpha,  // pointer to GPU memory
    float *img_red,             // pointer to GPU memory
    float *img_green,           // pointer to GPU memory
    float *img_blue,            // pointer to GPU memory
    GpuMemoryPool &memory_pool) {

    uint32_t threads_c = THREADS_X * THREADS_Y;
    uint32_t blocks_c = CEIL_DIV(n_circle, threads_c);

    // Get circle coverage
    int32_t num_tiles = CEIL_DIV(width, TILE_SIZE) * CEIL_DIV(height, TILE_SIZE);
    size_t circle_map_size = (size_t)num_tiles * (size_t)n_circle * sizeof(uint8_t);

    uint8_t *circle_map = reinterpret_cast<uint8_t *>(memory_pool.alloc(circle_map_size));

    tile_coverage<<<blocks_c, threads_c>>>(
        width,
        height,
        n_circle,
        circle_x,
        circle_y,
        circle_radius,
        circle_map,
        num_tiles);

    size_t scan_size = get_workspace_size_scan(n_circle, THREADS_SCAN);

    uint32_t **tile_circle_idxs =
        reinterpret_cast<uint32_t **>(memory_pool.alloc(num_tiles * sizeof(uint32_t *)));

    uint32_t *num_circles_per_tile = reinterpret_cast<uint32_t *>(
        memory_pool.alloc(num_tiles * sizeof(uint32_t))); //[num_tiles];
    void *scan_workspace = memory_pool.alloc(scan_size);

    // size_t mtemp = 0;

    uint32_t *compacted_stream = reinterpret_cast<uint32_t *>(
        memory_pool.alloc(num_tiles * MAX_CIRCLES_PER_TILE * sizeof(uint32_t)));

    for (int i = 0; i < num_tiles; ++i) {

        uint32_t *scanned_circle_idxs = launch_scan(
            (size_t)n_circle,
            circle_map + (size_t)i * (size_t)n_circle,
            scan_workspace);

        compact_stream<<<blocks_c, threads_c>>>(
            n_circle,
            scanned_circle_idxs,
            compacted_stream + i * MAX_CIRCLES_PER_TILE,
            num_circles_per_tile + i);
    }

    dim3 threads_p = dim3(THREADS_X, THREADS_Y);
    dim3 blocks_p = dim3(CEIL_DIV(width, TILE_SIZE), CEIL_DIV(height, TILE_SIZE));

    size_t render_shmem = sizeof(float) * (CIRCLE_STATS * THREADS_X * THREADS_Y);
    // CUDA_CHECK(cudaFuncSetAttribute(
    //     render_pixels,
    //     cudaFuncAttributeMaxDynamicSharedMemorySize,
    //     render_shmem));

    render_pixels<<<blocks_p, threads_p, render_shmem>>>(
        num_circles_per_tile,
        n_circle,
        num_tiles,
        width,
        height,
        circle_x,
        circle_y,
        circle_radius,
        circle_red,
        circle_green,
        circle_blue,
        circle_alpha,
        img_red,
        img_green,
        img_blue,
        compacted_stream);
}

} // namespace circles_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

GpuMemoryPool::~GpuMemoryPool() {
    for (auto ptr : allocations_) {
        CUDA_CHECK(cudaFree(ptr));
    }
}

void *GpuMemoryPool::alloc(size_t size) {
    if (next_idx_ < allocations_.size()) {
        auto idx = next_idx_++;
        if (size > capacities_.at(idx)) {
            CUDA_CHECK(cudaFree(allocations_.at(idx)));
            CUDA_CHECK(cudaMalloc(&allocations_.at(idx), size));
            CUDA_CHECK(cudaMemset(allocations_.at(idx), 0, size));
            capacities_.at(idx) = size;
        }
        return allocations_.at(idx);
    } else {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        CUDA_CHECK(cudaMemset(ptr, 0, size));
        allocations_.push_back(ptr);
        capacities_.push_back(size);
        next_idx_++;
        return ptr;
    }
}

void GpuMemoryPool::reset() {
    next_idx_ = 0;
    for (int32_t i = 0; i < allocations_.size(); i++) {
        CUDA_CHECK(cudaMemset(allocations_.at(i), 0, capacities_.at(i)));
    }
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Scene {
    int32_t width;
    int32_t height;
    std::vector<float> circle_x;
    std::vector<float> circle_y;
    std::vector<float> circle_radius;
    std::vector<float> circle_red;
    std::vector<float> circle_green;
    std::vector<float> circle_blue;
    std::vector<float> circle_alpha;

    int32_t n_circle() const { return circle_x.size(); }
};

struct Image {
    int32_t width;
    int32_t height;
    std::vector<float> red;
    std::vector<float> green;
    std::vector<float> blue;
};

float max_abs_diff(Image const &a, Image const &b) {
    float max_diff = 0.0f;
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        float diff_red = std::abs(a.red.at(idx) - b.red.at(idx));
        float diff_green = std::abs(a.green.at(idx) - b.green.at(idx));
        float diff_blue = std::abs(a.blue.at(idx) - b.blue.at(idx));
        max_diff = std::max(max_diff, diff_red);
        max_diff = std::max(max_diff, diff_green);
        max_diff = std::max(max_diff, diff_blue);
    }
    return max_diff;
}

struct Results {
    bool correct;
    float max_abs_diff;
    Image image_expected;
    Image image_actual;
    double time_ms;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename T> struct GpuBuf {
    T *data;

    explicit GpuBuf(size_t n) { CUDA_CHECK(cudaMalloc(&data, n * sizeof(T))); }

    explicit GpuBuf(std::vector<T> const &host_data) {
        CUDA_CHECK(cudaMalloc(&data, host_data.size() * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(
            data,
            host_data.data(),
            host_data.size() * sizeof(T),
            cudaMemcpyHostToDevice));
    }

    ~GpuBuf() { CUDA_CHECK(cudaFree(data)); }
};

Results run_config(Mode mode, Scene const &scene) {
    auto img_expected = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    render_cpu(
        scene.width,
        scene.height,
        scene.n_circle(),
        scene.circle_x.data(),
        scene.circle_y.data(),
        scene.circle_radius.data(),
        scene.circle_red.data(),
        scene.circle_green.data(),
        scene.circle_blue.data(),
        scene.circle_alpha.data(),
        img_expected.red.data(),
        img_expected.green.data(),
        img_expected.blue.data());

    auto circle_x_gpu = GpuBuf<float>(scene.circle_x);
    auto circle_y_gpu = GpuBuf<float>(scene.circle_y);
    auto circle_radius_gpu = GpuBuf<float>(scene.circle_radius);
    auto circle_red_gpu = GpuBuf<float>(scene.circle_red);
    auto circle_green_gpu = GpuBuf<float>(scene.circle_green);
    auto circle_blue_gpu = GpuBuf<float>(scene.circle_blue);
    auto circle_alpha_gpu = GpuBuf<float>(scene.circle_alpha);
    auto img_red_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_green_gpu = GpuBuf<float>(scene.height * scene.width);
    auto img_blue_gpu = GpuBuf<float>(scene.height * scene.width);

    auto memory_pool = GpuMemoryPool();

    auto reset = [&]() {
        CUDA_CHECK(
            cudaMemset(img_red_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(cudaMemset(
            img_green_gpu.data,
            0,
            scene.height * scene.width * sizeof(float)));
        CUDA_CHECK(
            cudaMemset(img_blue_gpu.data, 0, scene.height * scene.width * sizeof(float)));
        memory_pool.reset();
    };

    auto f = [&]() {
        circles_gpu::launch_render(
            scene.width,
            scene.height,
            scene.n_circle(),
            circle_x_gpu.data,
            circle_y_gpu.data,
            circle_radius_gpu.data,
            circle_red_gpu.data,
            circle_green_gpu.data,
            circle_blue_gpu.data,
            circle_alpha_gpu.data,
            img_red_gpu.data,
            img_green_gpu.data,
            img_blue_gpu.data,
            memory_pool);
    };

    reset();
    f();

    auto img_actual = Image{
        scene.width,
        scene.height,
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f),
        std::vector<float>(scene.height * scene.width, 0.0f)};

    CUDA_CHECK(cudaMemcpy(
        img_actual.red.data(),
        img_red_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.green.data(),
        img_green_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        img_actual.blue.data(),
        img_blue_gpu.data,
        scene.height * scene.width * sizeof(float),
        cudaMemcpyDeviceToHost));

    float max_diff = max_abs_diff(img_expected, img_actual);

    if (max_diff > 5e-2) {
        return Results{
            false,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    if (mode == Mode::TEST) {
        return Results{
            true,
            max_diff,
            std::move(img_expected),
            std::move(img_actual),
            0.0,
        };
    }

    double time_ms = benchmark_ms(1000.0, reset, f);

    return Results{
        true,
        max_diff,
        std::move(img_expected),
        std::move(img_actual),
        time_ms,
    };
}

template <typename Rng>
Scene gen_random(Rng &rng, int32_t width, int32_t height, int32_t n_circle) {
    auto unif_0_1 = std::uniform_real_distribution<float>(0.0f, 1.0f);
    auto z_values = std::vector<float>();
    for (int32_t i = 0; i < n_circle; i++) {
        float z;
        for (;;) {
            z = unif_0_1(rng);
            z = std::max(z, unif_0_1(rng));
            if (z > 0.01) {
                break;
            }
        }
        // float z = std::max(unif_0_1(rng), unif_0_1(rng));
        z_values.push_back(z);
    }
    std::sort(z_values.begin(), z_values.end(), std::greater<float>());

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };
    auto color_idx_dist = std::uniform_int_distribution<int>(0, colors.size() - 1);
    auto alpha_dist = std::uniform_real_distribution<float>(0.0f, 0.3f);

    int32_t fog_interval = n_circle / 10;
    float fog_alpha = 0.2;

    auto scene = Scene{width, height};
    float base_radius_scale = 1.0f;
    int32_t i = 0;
    for (float z : z_values) {
        float max_radius = base_radius_scale / z;
        float radius = std::max(1.0f, unif_0_1(rng) * max_radius);
        float x = unif_0_1(rng) * (width + 2 * max_radius) - max_radius;
        float y = unif_0_1(rng) * (height + 2 * max_radius) - max_radius;
        int color_idx = color_idx_dist(rng);
        uint32_t color = colors[color_idx];
        scene.circle_x.push_back(x);
        scene.circle_y.push_back(y);
        scene.circle_radius.push_back(radius);
        scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
        scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
        scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
        scene.circle_alpha.push_back(alpha_dist(rng));
        i++;
        if (i % fog_interval == 0 && i + 1 < n_circle) {
            scene.circle_x.push_back(float(width - 1) / 2.0f);
            scene.circle_y.push_back(float(height - 1) / 2.0f);
            scene.circle_radius.push_back(float(std::max(width, height)));
            scene.circle_red.push_back(1.0f);
            scene.circle_green.push_back(1.0f);
            scene.circle_blue.push_back(1.0f);
            scene.circle_alpha.push_back(fog_alpha);
        }
    }

    return scene;
}

constexpr float PI = 3.14159265359f;

Scene gen_overlapping_opaque() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    auto colors = std::vector<uint32_t>{
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    };

    int32_t n_circle = 20;
    int32_t n_ring = 4;
    float angle_range = PI;
    for (int32_t ring = 0; ring < n_ring; ring++) {
        float dist = 20.0f * (ring + 1);
        float saturation = float(ring + 1) / n_ring;
        float hue_shift = float(ring) / (n_ring - 1);
        for (int32_t i = 0; i < n_circle; i++) {
            float theta = angle_range * i / (n_circle - 1);
            float x = width / 2.0f - dist * std::cos(theta);
            float y = height / 2.0f - dist * std::sin(theta);
            scene.circle_x.push_back(x);
            scene.circle_y.push_back(y);
            scene.circle_radius.push_back(16.0f);
            auto color = colors[(i + ring * 2) % colors.size()];
            scene.circle_red.push_back(float((color >> 16) & 0xff) / 255.0f);
            scene.circle_green.push_back(float((color >> 8) & 0xff) / 255.0f);
            scene.circle_blue.push_back(float(color & 0xff) / 255.0f);
            scene.circle_alpha.push_back(1.0f);
        }
    }

    return scene;
}

Scene gen_overlapping_transparent() {
    int32_t width = 256;
    int32_t height = 256;

    auto scene = Scene{width, height};

    float offset = 20.0f;
    float radius = 40.0f;
    scene.circle_x = std::vector<float>{
        (width - 1) / 2.0f - offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f + offset,
        (width - 1) / 2.0f - offset,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
    };
    scene.circle_radius = std::vector<float>{
        radius,
        radius,
        radius,
        radius,
    };
    // 0xd32360
    // 0x2874aa
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0x28) / 255.0f,
        float(0x28) / 255.0f,
        float(0xd3) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x74) / 255.0f,
        float(0x74) / 255.0f,
        float(0x23) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0xaa) / 255.0f,
        float(0xaa) / 255.0f,
        float(0x60) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        0.75f,
        0.75f,
        0.75f,
        0.75f,
    };
    return scene;
}

Scene gen_simple() {
    /*
        0xd32360,
        0xcc9f26,
        0x208020,
        0x2874aa,
    */
    int32_t width = 256;
    int32_t height = 256;
    auto scene = Scene{width, height};
    scene.circle_x = std::vector<float>{
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
        (width - 1) * 0.25f,
        (width - 1) * 0.75f,
    };
    scene.circle_y = std::vector<float>{
        (height - 1) * 0.25f,
        (height - 1) * 0.25f,
        (height - 1) * 0.75f,
        (height - 1) * 0.75f,
    };
    scene.circle_radius = std::vector<float>{
        40.0f,
        40.0f,
        40.0f,
        40.0f,
    };
    scene.circle_red = std::vector<float>{
        float(0xd3) / 255.0f,
        float(0xcc) / 255.0f,
        float(0x20) / 255.0f,
        float(0x28) / 255.0f,
    };
    scene.circle_green = std::vector<float>{
        float(0x23) / 255.0f,
        float(0x9f) / 255.0f,
        float(0x80) / 255.0f,
        float(0x74) / 255.0f,
    };
    scene.circle_blue = std::vector<float>{
        float(0x60) / 255.0f,
        float(0x26) / 255.0f,
        float(0x20) / 255.0f,
        float(0xaa) / 255.0f,
    };
    scene.circle_alpha = std::vector<float>{
        1.0f,
        1.0f,
        1.0f,
        1.0f,
    };
    return scene;
}

// Output image writers: BMP file header structure
#pragma pack(push, 1)
struct BMPHeader {
    uint16_t fileType{0x4D42};   // File type, always "BM"
    uint32_t fileSize{0};        // Size of the file in bytes
    uint16_t reserved1{0};       // Always 0
    uint16_t reserved2{0};       // Always 0
    uint32_t dataOffset{54};     // Start position of pixel data
    uint32_t headerSize{40};     // Size of this header (40 bytes)
    int32_t width{0};            // Image width in pixels
    int32_t height{0};           // Image height in pixels
    uint16_t planes{1};          // Number of color planes
    uint16_t bitsPerPixel{24};   // Bits per pixel (24 for RGB)
    uint32_t compression{0};     // Compression method (0 for uncompressed)
    uint32_t imageSize{0};       // Size of raw bitmap data
    int32_t xPixelsPerMeter{0};  // Horizontal resolution
    int32_t yPixelsPerMeter{0};  // Vertical resolution
    uint32_t colorsUsed{0};      // Number of colors in the color palette
    uint32_t importantColors{0}; // Number of important colors
};
#pragma pack(pop)

void write_bmp(
    std::string const &fname,
    uint32_t width,
    uint32_t height,
    const std::vector<uint8_t> &pixels) {
    BMPHeader header;
    header.width = width;
    header.height = height;

    uint32_t rowSize = (width * 3 + 3) & (~3); // Align to 4 bytes
    header.imageSize = rowSize * height;
    header.fileSize = header.dataOffset + header.imageSize;

    std::ofstream file(fname, std::ios::binary);
    file.write(reinterpret_cast<const char *>(&header), sizeof(header));

    // Write pixel data with padding
    std::vector<uint8_t> padding(rowSize - width * 3, 0);
    for (int32_t idx_y = height - 1; idx_y >= 0;
         --idx_y) { // BMP stores pixels from bottom to top
        const uint8_t *row = &pixels[idx_y * width * 3];
        file.write(reinterpret_cast<const char *>(row), width * 3);
        if (!padding.empty()) {
            file.write(reinterpret_cast<const char *>(padding.data()), padding.size());
        }
    }
}

uint8_t float_to_byte(float x) {
    if (x < 0) {
        return 0;
    } else if (x >= 1) {
        return 255;
    } else {
        return x * 255.0f;
    }
}

void write_image(std::string const &fname, Image const &img) {
    auto pixels = std::vector<uint8_t>(img.width * img.height * 3);
    for (int32_t idx = 0; idx < img.width * img.height; idx++) {
        float red = img.red.at(idx);
        float green = img.green.at(idx);
        float blue = img.blue.at(idx);
        // BMP stores pixels in BGR order
        pixels.at(idx * 3) = float_to_byte(blue);
        pixels.at(idx * 3 + 1) = float_to_byte(green);
        pixels.at(idx * 3 + 2) = float_to_byte(red);
    }
    write_bmp(fname, img.width, img.height, pixels);
}

Image compute_img_diff(Image const &a, Image const &b) {
    auto img_diff = Image{
        a.width,
        a.height,
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
        std::vector<float>(a.height * a.width, 0.0f),
    };
    for (int32_t idx = 0; idx < a.width * a.height; idx++) {
        img_diff.red.at(idx) = std::abs(a.red.at(idx) - b.red.at(idx));
        img_diff.green.at(idx) = std::abs(a.green.at(idx) - b.green.at(idx));
        img_diff.blue.at(idx) = std::abs(a.blue.at(idx) - b.blue.at(idx));
    }
    return img_diff;
}

struct SceneTest {
    std::string name;
    Mode mode;
    Scene scene;
};

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto scenes = std::vector<SceneTest>();
    scenes.push_back({"simple", Mode::TEST, gen_simple()});
    scenes.push_back({"overlapping_opaque", Mode::TEST, gen_overlapping_opaque()});
    scenes.push_back(
        {"overlapping_transparent", Mode::TEST, gen_overlapping_transparent()});
    scenes.push_back(
        {"ten_million_circles",
         Mode::BENCHMARK,
         gen_random(rng, 1024, 1024, 10'000'000)});

    int32_t fail_count = 0;

    int32_t count = 0;
    for (auto const &scene_test : scenes) {
        auto i = count++;
        printf("\nTesting scene '%s'\n", scene_test.name.c_str());
        auto results = run_config(scene_test.mode, scene_test.scene);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_cpu.bmp",
            results.image_expected);
        write_image(
            std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                "_gpu.bmp",
            results.image_actual);
        if (!results.correct) {
            printf("  Result did not match expected image\n");
            printf("  Max absolute difference: %.2e\n", results.max_abs_diff);
            auto diff = compute_img_diff(results.image_expected, results.image_actual);
            write_image(
                std::string("out/img") + std::to_string(i) + "_" + scene_test.name +
                    "_diff.bmp",
                diff);
            printf(
                "  (Wrote image diff to 'out/img%d_%s_diff.bmp')\n",
                i,
                scene_test.name.c_str());
            fail_count++;
            continue;
        } else {
            printf("  OK\n");
        }
        if (scene_test.mode == Mode::BENCHMARK) {
            printf("  Time: %f ms\n", results.time_ms);
        }
    }

    if (fail_count) {
        printf("\nCorrectness: %d tests failed\n", fail_count);
    } else {
        printf("\nCorrectness: All tests passed\n");
    }

    return 0;
}
