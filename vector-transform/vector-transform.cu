#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void vectorTransformKernel(int height, int n  , float* A, float* B, float* Result) {
 unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
 unsigned j = blockIdx.y * blockDim.y + threadIdx.y;
 if(i < height && j < n) Result[j] = Result[j]+A[j]*B[j];

}

void vectorTransformCuda(int n, float* a, float* b, float* result) {
    int threadBlockSize = 512;
    int height = 5;
    // allocate the vectors on the GPU
    float* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n*height*sizeof(float)));
    if (deviceA == NULL) {
        cout << "could not allocate memory!" << endl;
        return;
    }
    float* deviceB = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceB,n*height*sizeof(float)));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        cout << "could not allocate memory!" << endl;
        return;
    }
    float* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n*height*sizeof(float)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        cout << "could not allocate memory!" << endl;
        return;
    }

    timer kernelTime1 = timer("kernelTime1");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceA, a, n*height*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*height*sizeof(float), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    dim3 grid(1,n/threadBlockSize);
    dim3 block(height,threadBlockSize);

    vectorTransformKernel<<<grid, block>>>(height,n,deviceA, deviceB, deviceResult);

    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(result, deviceResult, n*height*sizeof(float), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    cout << "vector-transform (kernel): \t\t" << kernelTime1  << endl;
    cout << "vector-transform (memory): \t\t" << memoryTime << endl;
}

int vectorTransformSeq(int n, float* a, float* b, float* result) {
  int i,j; 

  timer sequentialTime = timer("Sequential");
  
  sequentialTime.start();
  for (j=0; j<5; j++) {
    for (i=0; i<n; i++) {
	result[i] = result[i]+a[i]*b[i];
    }
  }
  sequentialTime.stop();
  
  cout << "vector-transform (sequential): \t\t" << sequentialTime << endl;

}

int main(int argc, char* argv[]) {
    int n = 655360;
    float* a = new float[n];
    float* b = new float[n];
    float* result = new float[n];
    float* result_s = new float[n];

    if (argc > 1) n = atoi(argv[1]);

    cout << "Iteratively transform vector A with vector B of " << n << " integer elements." << endl;
    // initialize the vectors.
    for(int i=0; i<n; i++) {
        a[i] = i;
        b[i] = 0.1*i;
	result[i]=0;
	result_s[i]=0;
    }

    vectorTransformSeq(n, a, b, result_s);
    vectorTransformCuda(n, a, b, result);
    
    // verify the resuls
    for(int i=0; i<n; i++) {
	  if (result[i]!=result_s[i]) {
      if (fabs(result[i] - result_s[i]) >0.001)
        cout << "error in results! Element " << i << " is " << result[i] << ", but should be " << result_s[i] << endl; 
            exit(1);
        }
    }
    cout << "results OK!" << endl;
            
    delete[] a;
    delete[] b;
    delete[] result;
    
    return 0;
}
