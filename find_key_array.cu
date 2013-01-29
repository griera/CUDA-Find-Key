#include <stdio.h>
#include <stdlib.h>

/*
 * See section "B. 19 Launch Bounds" from "CUDA C Programming Guide" for more
 * information about the optimal launch bounds, which differ across the major
 * architecture revisions
 */
#define THREADS_PER_BLOCK 256

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

// Host function that finds a specify key in array
void find_key_cpu(const int* array_in, int n, int key);

// Variable used as a parameter of atomic function
// It resides in global memory
__device__ unsigned int first_occur;

// Variable used to know how many thread blocks have finished their tasks
// It resides in global memory
__device__ unsigned int count = 0;

// Variable used to control when all blocks have finished their tasks
// It resides in shared memory
__shared__ bool all_blocks_finished;

// Kernel code
__global__ void find_key_gpu(const int* array_in, int n, int key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only one thread of the whole device initializes the variable
    // and make sure sure its result is visible to all other threads
    if (idx == 0) {
        first_occur = n;
        __threadfence();
    }

    // Avoid wrong accesses to array_in if there are more threads than elements
    if (idx < n) {
        int found = (array_in[idx] == key) ? idx : n;
        atomicMin(&first_occur, found);

        // Thread 0 of each block signals that its work has been finisehd
        if (threadIdx.x == 0) {
            unsigned int value = atomicInc(&count, gridDim.x);
            all_blocks_finished = (value == (gridDim.x - 1));
        }

        // Synchronize to make sure that each thread
        // reads the correct value of all_blocks_finished
        __syncthreads();

        if (all_blocks_finished) {
            if (threadIdx.x == 0) {

                // Thread 0 of last block is responsible to print the final result
                // Only one thread in the whole device
                if (first_occur < n) {
                    printf("The first ocurrence of key (%d) has been found at position %d\n"
                            , key, first_occur);
                }
                else {
                    printf("The key (%d) is not found\n", key);
                }
            }
        }
    }
}

// Kernel code
__global__ void find_key_gpu_fast(const int* array_in, int n, int key) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid wrong accesses to array_in if there are more threads than elements
    if (idx < n) {
        if (idx == 0) {
            first_occur = n;
            __threadfence();
        }
        if (array_in[idx] == key) {
            atomicMin(&first_occur, idx);
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s size key\n", argv[0]);
        fprintf(stderr, "       size is the number of input array's elements\n");
        fprintf(stderr, "       key is the number which we want to find\n");
        return EXIT_FAILURE;
    }

    cudaEvent_t start, stop;
    float elapsed_time_ms;

    int key = atoi(argv[2]);

    // Pointer for host memory and size
    int *h_array;
    unsigned int num_elems = atoi(argv[1]);
    size_t array_size = num_elems * sizeof(int);

    // pointer for device memory
    int *dev_array_in;

    // allocate host and device memory
    h_array = (int *) malloc(array_size);
    cudaMalloc(&dev_array_in, array_size);

    // Check for any CUDA errors
    checkCUDAError("cudaMalloc");

    /*
     * Also host memory allocation can be done using cudaMallocHost
     * cudaMallocHost(&h_array, array_size);
     *
     * Or also cudaHostAlloc
     * cudaHostAlloc(&h_array, array_size, cudaHostAllocDefault);
     */

    // Initialize host memory
    for (unsigned int i = 0; i < num_elems; ++i) {
        h_array[i] = i;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /*------------------------ COMPUTATION ON CPU ----------------------------*/
    cudaEventRecord(start, 0);
    // cudaEventSynchronize(start); needed?

    find_key_cpu(h_array, num_elems, key);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    printf("Time to find key on CPU: %f ms.\n\n", elapsed_time_ms);

    /*--------------- COMPUTATION ON GPU (find_key() kernel) -----------------*/

    // Host to device memory copy
    cudaMemcpy(dev_array_in, h_array, array_size, cudaMemcpyHostToDevice);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpy");

    // Set grid and block dimensions properly
    // num_elems + (TREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK is equal to call
    // ceil(num_elems/THREADS_PER_BLOCK) function from C Math Library
    int blocks_per_grid = (int) (num_elems + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEventRecord(start, 0);
    // cudaEventSynchronize(start); needed?

    // Launch kernel
    find_key_gpu<<<blocks_per_grid, THREADS_PER_BLOCK>>>(dev_array_in, num_elems, key);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    // Check for any CUDA errors
    checkCUDAError("kernel invocation");

    printf("Time to find key on GPU: %f ms.\n\n", elapsed_time_ms);

    // Block until the device has completed their tasks
    //cudaDeviceSynchronize();

    /*-------------- COMPUTATION ON GPU (find_key_fast() kernel) -------------*/

    cudaEventRecord(start, 0);
    // cudaEventSynchronize(start); needed?

    // Launch kernel
    find_key_gpu_fast<<<blocks_per_grid, THREADS_PER_BLOCK>>>(dev_array_in, num_elems, key);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);

    // Check for any CUDA errors
    checkCUDAError("kernel invocation");

    // Get the final result from global memory (on device)
    // and copy it to h_first_occur pointer (host memory)
    unsigned int h_first_occur;
    cudaMemcpyFromSymbol(&h_first_occur, first_occur, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);

    // Check for any CUDA errors
    checkCUDAError("cudaMemcpyFromSymbol");

    if (h_first_occur < num_elems) {
        printf("The first ocurrence of key (%d) has been found at position %d\n"
                , key, h_first_occur);
    }
    else {
        printf("The key (%d) is not found\n", key);
    }

    printf("Time to find key on GPU (fast kernel version): %f ms.\n\n", elapsed_time_ms);

    // Block until the device has completed their tasks
    //cudaDeviceSynchronize();

    // Free device and host memory
    cudaFree(dev_array_in);
    free(h_array);

    // Check for any CUDA errors
    checkCUDAError("cudaFree");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void find_key_cpu(const int* array_in, int n, int key) {
    bool found = false;
    for (unsigned int i = 0; !found && i < n; ++i) {
        if (array_in[i] == key) {
            found = true;
            printf("The first ocurrence of key (%d) has been found at position %d\n"
                    , key, i);
        }
    }
    if (!found) {
        printf("The key (%d) is not found\n", key);
    }
}
