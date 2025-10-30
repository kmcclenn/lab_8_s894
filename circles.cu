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

/* TODO: your GPU kernels here... */

__global__ void reduce_uint8(size_t n, uint8_t const *x, uint32_t *out) {

    uint8_t const *x_block = x + blockIdx.x * blockDim.x;

    extern __shared__ __align__(16) char shmem_raw[];
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    shmem[SHMEM_PADDING(threadIdx.x)] =
        *reinterpret_cast<const uint32_t *>(&x_block[threadIdx.x]);

    __syncthreads();

    for (int i = 1; i < THREADS_SCAN; i <<= 1) {
        uint32_t add = (threadIdx.x >= i) ? shmem[SHMEM_PADDING(threadIdx.x - i)] : 0;
        __syncthreads();
        shmem[SHMEM_PADDING(threadIdx.x)] = add + shmem[SHMEM_PADDING(threadIdx.x)];
        __syncthreads();
    }

    int last = MIN((int)n - (int)(blockIdx.x * blockDim.x), (int)blockDim.x) - 1;
    out[blockIdx.x] = shmem[SHMEM_PADDING(last)];
}

__global__ void reduce(size_t n, uint32_t const *x, uint32_t *out) {

    uint32_t const *x_block = x + blockIdx.x * blockDim.x;

    extern __shared__ __align__(16) char shmem_raw[];
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    shmem[SHMEM_PADDING(threadIdx.x)] = x_block[threadIdx.x];

    __syncthreads();

    for (int i = 1; i < THREADS_SCAN; i <<= 1) {
        uint32_t add = (threadIdx.x >= i) ? shmem[SHMEM_PADDING(threadIdx.x - i)] : 0;
        __syncthreads();
        shmem[SHMEM_PADDING(threadIdx.x)] = add + shmem[SHMEM_PADDING(threadIdx.x)];
        __syncthreads();
    }

    int last = MIN((int)n - (int)(blockIdx.x * blockDim.x), (int)blockDim.x) - 1;
    out[blockIdx.x] = shmem[SHMEM_PADDING(last)];
}

__global__ void scan_block(size_t n, uint32_t const *x, uint32_t *out) {
    uint32_t accumulator = 0;

    extern __shared__ __align__(16) char shmem_raw[];
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    shmem[SHMEM_PADDING(threadIdx.x)] = x[threadIdx.x];

    __syncthreads();

    for (int i = 1; i < THREADS_SCAN; i <<= 1) {
        uint32_t add = (threadIdx.x >= i) ? shmem[SHMEM_PADDING(threadIdx.x - i)] : 0;
        __syncthreads();
        shmem[SHMEM_PADDING(threadIdx.x)] = add + shmem[SHMEM_PADDING(threadIdx.x)];
        __syncthreads();
    }
    if (threadIdx.x < n)
        out[threadIdx.x] = shmem[SHMEM_PADDING(threadIdx.x)];
}

__global__ void
scan(size_t n, uint32_t const *x, uint32_t const *end_points, uint32_t *out) {

    extern __shared__ __align__(16) char shmem_raw[];
    uint32_t *shmem = reinterpret_cast<uint32_t *>(shmem_raw);

    uint32_t const *x_block = x + blockIdx.x * blockDim.x;
    uint32_t *out_block = out + blockIdx.x * blockDim.x;

    shmem[SHMEM_PADDING(threadIdx.x)] = x_block[threadIdx.x];

    __syncthreads();

    for (int i = 1; i < THREADS_SCAN; i <<= 1) {
        uint32_t add = (threadIdx.x >= i) ? shmem[SHMEM_PADDING(threadIdx.x - i)] : 0;
        __syncthreads();
        shmem[SHMEM_PADDING(threadIdx.x)] = add + shmem[SHMEM_PADDING(threadIdx.x)];
        __syncthreads();
    }
    uint32_t block_carry = (blockIdx.x == 0) ? 0 : end_points[blockIdx.x - 1];
    out_block[threadIdx.x] = block_carry + shmem[SHMEM_PADDING(threadIdx.x)];
}

// Returns desired size of scratch buffer in bytes.
size_t get_workspace_size_scan(size_t n) {
    /* TODO: your CPU code here... */
    size_t total = n;
    size_t size = n;
    while (size > THREADS_SCAN) {
        total += size;
        size = CEIL_DIV(size, THREADS_SCAN);
    }

    return 2 * total * sizeof(uint32_t);
}

__global__ void fill_data(uint8_t *data_uint8, uint32_t *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = *reinterpret_cast<uint32_t *>(&data_uint8[idx]);
}

// 'launch_scan'
//

uint32_t *launch_scan(
    size_t n,
    uint8_t *x,     // pointer to GPU memory
    void *workspace // pointer to GPU memory
) {
    /* TODO: your CPU code here... */
    uint32_t *arr = reinterpret_cast<uint32_t *>(workspace); // size n
    uint8_t *data_uint8 = x;
    uint32_t *data;
    uint32_t *orig_x;
    // compute sums per block

    size_t size = n;
    size_t offsets[8];
    offsets[0] = 0;
    int iter = 1;

    // first iteration with uint8
    if (size > THREADS_SCAN) {
        size_t blocks = CEIL_DIV(size, THREADS_SCAN);
        reduce_uint8<<<blocks, THREADS_SCAN, (THREADS_SCAN + PAD) * sizeof(uint32_t)>>>(
            size,
            data_uint8,
            arr + offsets[iter - 1]);

        size = blocks;
        data = arr + offsets[iter - 1];
        offsets[iter] = offsets[iter - 1] + size;
        iter++;
    } else if (threadIdx.x < n) {
        fill_data<<<CEIL_DIV(size, THREADS_SCAN), THREADS_SCAN>>>(data_uint8, orig_x);
        data = orig_x;
    }

    while (size > THREADS_SCAN) {
        size_t blocks = CEIL_DIV(size, THREADS_SCAN);
        reduce<<<blocks, THREADS_SCAN, (THREADS_SCAN + PAD) * sizeof(uint32_t)>>>(
            size,
            data,
            arr + offsets[iter - 1]);

        size = blocks;
        data = arr + offsets[iter - 1];

        offsets[iter] = offsets[iter - 1] + size;
        iter++;
    }
    iter--;

    uint32_t *final_block = (iter == 0) ? orig_x : arr + offsets[iter];
    size_t threads = MIN(THREADS_SCAN, n);

    uint32_t *base_out = (iter == 0) ? arr : final_block + threads;

    scan_block<<<1, threads, (threads + PAD) * sizeof(uint32_t)>>>(
        size,
        final_block,
        base_out);

    size_t larger_size = threads;
    base_out += larger_size;
    uint32_t *end_points = arr + offsets[iter];
    if (n > THREADS_SCAN) {
        while (iter >= 0) {

            larger_size = (iter == 0) ? n : (offsets[iter] - offsets[iter - 1]);
            size_t blocks = CEIL_DIV(larger_size, THREADS_SCAN);

            uint32_t *data_ptr = (iter == 0) ? orig_x : arr + offsets[iter - 1];
            scan<<<blocks, THREADS_SCAN, (THREADS_SCAN + PAD) * sizeof(uint32_t)>>>(
                larger_size,
                data_ptr,
                end_points,
                base_out);

            size = larger_size;
            iter--;
            end_points = base_out;

            base_out += larger_size;
        }
    }
    base_out -= larger_size;

    return base_out;
}

#define THREADS_X 32
#define THREADS_Y 32
#define TILE_SIZE 32

/* TODO: your GPU kernels here... */
__global__ void initialize_background(
    int32_t width,
    int32_t height,
    float *img_red,
    float *img_green,
    float *img_blue) {

    int32_t pixel_row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t pixel_idx = pixel_row * width + pixel_col;

    if (pixel_idx < width * height) {
        img_red[pixel_idx] = 1.0f;
        img_green[pixel_idx] = 1.0f;
        img_blue[pixel_idx] = 1.0f;
    }
}

__global__ void tile_coverage(
    int32_t width,
    int32_t height,
    int32_t n_circle,
    float const *circle_x,
    float const *circle_y,
    float const *circle_radius,
    uint8_t *circle_map) {

    int32_t circle = blockIdx.x * blockDim.x + threadIdx.x;
    // uint8_t *tile_circle_map =
    //     circle_map + tile_num * n_circle; // TODO: update if 8 circles/byte

    int tile_width = width / TILE_SIZE;
    int tile_height = height / TILE_SIZE;

    float x = circle_x[circle];
    float y = circle_y[circle];
    float rad = circle_radius[circle];

    float left = x - rad;
    float right = x + rad;
    float top = y - rad;
    float bot = y + rad;

    // go through each corner of the bounding box
    int tile_tl = -1;
    int tile_bl = -1;
    int tile_tr = -1;
    int tile_br = -1;

    if (left > 0 && top > 0) {
        tile_tl = (left / TILE_SIZE) * tile_width + (top / TILE_SIZE);
    }
    if (left > 0 && bot < height - 1) {
        tile_bl = (left / TILE_SIZE) * tile_width + (top / TILE_SIZE);
    }
    if (right < width - 1 && top > 0) {
        tile_tr = (right / TILE_SIZE) * tile_width + (top / TILE_SIZE);
    }
    if (right < width - 1 && bot < height - 1) {
        tile_br = (right / TILE_SIZE) * tile_width + (bot / TILE_SIZE);
    }

    (circle_map + tile_tl)[circle] = 1;
    (circle_map + tile_bl)[circle] = 1;
    (circle_map + tile_tr)[circle] = 1;
    (circle_map + tile_br)[circle] = 1;
}

__global__ void compact_stream(uint32_t *scanned_idxs, uint32_t *compacted_idxs) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        compacted_idxs[0] = scanned_idxs[0] != 0;
    } else if (scanned_idxs[idx] != scanned_idxs[idx - 1]) {
        compacted_idxs[idx] = scanned_idxs[idx];
    }
}

__global__ void render_pixels(
    uint32_t **tile_circle_idxs,
    uint32_t *num_circles_per_tile,
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

    int32_t tile_idx = blockIdx.y * TILE_SIZE + blockIdx.x;
    int32_t pixel_row = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t pixel_col = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t pixel_idx = pixel_row * width + pixel_col;

    uint32_t *tile_circles = tile_circle_idxs[tile_idx];
    uint32_t n_t_circles = num_circles_per_tile[tile_idx];

    for (int i = 0; i < n_t_circles; ++i) {
        uint32_t c_idx = tile_circles[i];

        float x = circle_x[c_idx];
        float y = circle_y[c_idx];
        float rad = circle_radius[c_idx];

        float dy = pixel_row - y;
        float dx = pixel_col - x;

        if (dy * dy + dx * dx < rad * rad) {

            float red = img_red[pixel_idx];
            float green = img_green[pixel_idx];
            float blue = img_blue[pixel_idx];

            float alpha = circle_alpha[i];

            red = circle_red[i] * alpha + red * (1.0f - alpha);
            green = circle_green[i] * alpha + green * (1.0f - alpha);
            blue = circle_blue[i] * alpha + blue * (1.0f - alpha);

            img_red[pixel_idx] = red;
            img_green[pixel_idx] = green;
            img_blue[pixel_idx] = blue;
        }
    }
}

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

    // Initialize background to white
    dim3 threads_p = dim3(THREADS_X, THREADS_Y);
    dim3 blocks_p = dim3(CEIL_DIV(width, THREADS_X), CEIL_DIV(height, THREADS_Y));

    uint32_t threads_c = THREADS_X * THREADS_Y;
    uint32_t blocks_c = CEIL_DIV(n_circle, threads_c);

    initialize_background<<<blocks_p, threads_p>>>(
        width,
        height,
        img_red,
        img_green,
        img_blue);

    // Get circle coverage
    int32_t num_tiles = (width / TILE_SIZE) * (height / TILE_SIZE);
    size_t circle_map_size = num_tiles * n_circle *
        sizeof(uint8_t); // TODO: update because can store 8 flags in a char
    uint8_t *circle_map = reinterpret_cast<uint8_t *>(memory_pool.alloc(circle_map_size));

    // dim3 threads = dim3(TILE_SIZE, TILE_SIZE);
    // dim3 blocks = dim3(CEIL_DIV(width, TILE_SIZE), CEIL_DIV(height, TILE_SIZE));
    tile_coverage<<<blocks_c, threads_c>>>(
        width,
        height,
        n_circle,
        circle_x,
        circle_y,
        circle_radius,
        circle_map);

    size_t scan_size = get_workspace_size_scan(n_circle);

    uint32_t *tile_circle_idxs[num_tiles];
    uint32_t *num_circles_per_tile;
    for (int i = 0; i < num_tiles; ++i) {
        void *scan_workspace = memory_pool.alloc(scan_size);
        uint32_t *scanned_circle_idxs =
            launch_scan(n_circle, circle_map + i * n_circle, scan_workspace);

        num_circles_per_tile[i] = scanned_circle_idxs[n_circle - 1];
        uint32_t *compacted_stream = reinterpret_cast<uint32_t *>(
            memory_pool.alloc(num_circles_per_tile[i] * sizeof(uint32_t)));

        compact_stream<<<blocks_c, threads_c>>>(scanned_circle_idxs, compacted_stream);
        tile_circle_idxs[i] = compacted_stream;
    }

    // parallelize over pixels
    render_pixels<<<blocks_p, threads_p>>>(
        tile_circle_idxs,
        num_circles_per_tile,
        width,
        height,
        n_circle,
        circle_x,
        circle_y,
        circle_radius,
        circle_red,
        circle_green,
        circle_blue,
        circle_alpha,
        img_red,
        img_green,
        img_blue);
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
