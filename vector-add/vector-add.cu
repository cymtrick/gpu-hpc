#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <chrono>

using namespace std::chrono;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "cuda error: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

__global__ void vectorAddKernel(int n,float* A, float* B, float* Result) {
    // insert operation here
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<n)  Result[i]= A[i] + B[i];
}

__global__ void vectorSubKernel(int n,float* A, float* B, float* Result) {
    // insert operation here
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<n)  Result[i]= A[i] - B[i];
}

__global__ void vectorMulKernel(int n,float* A, float* B, float* Result) {
    // insert operation here
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<n)  Result[i]= A[i] * B[i];
}

__global__ void vectorDivKernel(int n,float* A, float* B, float* Result) {
    // insert operation here
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i<n)  Result[i]= A[i] / B[i];
}



void vectorOpsCuda(int n,int operation,int gridGeoThreadBlock, float* a, float* b, float* result) {
    int threadBlockSize = 512;

    if (gridGeoThreadBlock!=0) threadBlockSize = gridGeoThreadBlock;

    // allocate the vectors on the GPU
    float* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(float)));
    if (deviceA == NULL) {
        std::cout << "could not allocate memory!" << std::endl;
        return;
    }
    float* deviceB = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceB, n * sizeof(float)));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        std::cout << "could not allocate memory!" << std::endl;
        return;
    }
    float* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(float)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        std::cout << "could not allocate memory!" << std::endl;
        return;
    }

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(float), cudaMemcpyHostToDevice));

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    // execute kernel
    if(operation == 0) vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(n, deviceA, deviceB, deviceResult);
    if(operation == 1) vectorSubKernel<<<n/threadBlockSize, threadBlockSize>>>(n, deviceA, deviceB, deviceResult);
    if(operation == 2) vectorMulKernel<<<n/threadBlockSize, threadBlockSize>>>(n, deviceA, deviceB, deviceResult);
    if(operation == 3) vectorDivKernel<<<n/threadBlockSize, threadBlockSize>>>(n, deviceA, deviceB, deviceResult);
    cudaDeviceSynchronize();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    high_resolution_clock::time_point t3 = high_resolution_clock::now();

    // copy result back
    checkCudaCall(cudaMemcpy(result, deviceResult, n * sizeof(float), cudaMemcpyDeviceToHost));

    high_resolution_clock::time_point t4 = high_resolution_clock::now();

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    if(operation == 0) std::cout << "vector-add (H2D):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
    if(operation == 0) std::cout << "vector-add (kernel): \t\t" << duration_cast<microseconds>(t3 - t2).count() << "us" << std::endl;
    if(operation == 0) std::cout << "vector-add (D2H):    \t\t" << duration_cast<microseconds>(t4 - t3).count() << "us" << std::endl;

    if(operation == 1) std::cout << "vector-sub (H2D):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
    if(operation == 1) std::cout << "vector-sub (kernel): \t\t" << duration_cast<microseconds>(t3 - t2).count() << "us" << std::endl;
    if(operation == 1) std::cout << "vector-sub (D2H):    \t\t" << duration_cast<microseconds>(t4 - t3).count() << "us" << std::endl;

    if(operation == 2) std::cout << "vector-mul (H2D):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
    if(operation == 2) std::cout << "vector-mul (kernel): \t\t" << duration_cast<microseconds>(t3 - t2).count() << "us" << std::endl;
    if(operation == 2) std::cout << "vector-mul (D2H):    \t\t" << duration_cast<microseconds>(t4 - t3).count() << "us" << std::endl;

    if(operation == 3) std::cout << "vector-div (H2D):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
    if(operation == 3) std::cout << "vector-div (kernel): \t\t" << duration_cast<microseconds>(t3 - t2).count() << "us" << std::endl;
    if(operation == 3) std::cout << "vector-div (D2H):    \t\t" << duration_cast<microseconds>(t4 - t3).count() << "us" << std::endl;
}


void vectorOpsSeq(int n, int operation, float* a, float* b, float* result) {
    int i;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    if(operation == 0) for (i=0; i<n; i++) { result[i] = a[i]+b[i]; } 
    if(operation == 1) for (i=0; i<n; i++) { result[i] = a[i]-b[i]; } 
    if(operation == 2) for (i=0; i<n; i++) { result[i] = a[i]*b[i]; } 
    if(operation == 3) for (i=0; i<n; i++) { result[i] = a[i]/b[i]; } 

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    if(operation == 0)  std::cout << "vector-add (seq):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
    if(operation == 1)  std::cout << "vector-sub (seq):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
    if(operation == 2)  std::cout << "vector-mul (seq):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;
    if(operation == 3)  std::cout << "vector-div (seq):    \t\t" << duration_cast<microseconds>(t2 - t1).count() << "us" << std::endl;


    
    
}

int main(int argc, char* argv[]) {
    int n = 655360;
    if (argc > 1) n = atoi(argv[1]);
    float* a = (float*)malloc(n * sizeof(float));
    float* b = (float*)malloc(n * sizeof(float));
    float* result = (float*)malloc(n * sizeof(float));
    float* result_s = (float*)malloc(n * sizeof(float));
    int gridGeoThreadBlock = 0;
    int operation = 0;
    if (argc > 2) gridGeoThreadBlock = atoi(argv[2]);
    if (argc > 3) operation = atoi(argv[3]);

    if(operation == 0) std::cout << "Adding two vectors of " << n << " integer elements." << std::endl;
    if(operation == 1) std::cout << "Subtract two vectors of " << n << " integer elements." << std::endl;
    if(operation == 2) std::cout << "Multiply two vectors of " << n << " integer elements." << std::endl;
    if(operation == 3) std::cout << "Divison two vectors of " << n << " integer elements." << std::endl; 

    // initialize the vectors.
    for(long int i=0; i<n; i++) {
        a[i] = i;
        b[i] = i;
    }

    vectorOpsSeq(n,operation, a, b, result_s);

    vectorOpsCuda(n, operation, gridGeoThreadBlock, a, b, result);

    // verify the resuls
    for(int i=0; i<n; i++) {
        if (result[i] != result_s[i]) {
            std::cout << "error in results! Element " << i << " is " << result[i] << ", but should be " << result_s[i] << std::endl;
            exit(1);
        }
    }

    std::cout << "results OK!" << std::endl;

    delete[] a;
    delete[] b;
    delete[] result;
    delete[] result_s;

    return 0;
}