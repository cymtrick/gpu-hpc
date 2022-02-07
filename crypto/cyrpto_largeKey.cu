#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;

char* deviceDataIn;
char* deviceDataOut;
int* largeKeyDevice;

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


__global__ void encryptKernel(int n, int largeKeySize, char* deviceDataIn, char* deviceDataOut, int* largeKeyDevice) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int key_index = index % largeKeySize; 
    int key = largeKeyDevice[key_index];
    if(index<=n) {
        if (deviceDataIn[index] >= 32) {
            if (deviceDataIn[index] + key >= 127) {
                deviceDataOut[index] = ((deviceDataIn[index] + key) % 127) + 32;
            } else {
                deviceDataOut[index]= deviceDataIn[index] + key;
            }
        } else {
            deviceDataOut[index] = deviceDataIn[index];
        }
    }
}


__global__ void decryptKernel(int n, int largeKeySize, char* deviceDataIn, char* deviceDataOut, int* largeKeyDevice) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int key_index = index % largeKeySize; 
    int key = largeKeyDevice[key_index];
    if(index<=n) {
        if (deviceDataIn[index] >= 32) {
            if (deviceDataIn[index] - key < 32) {
                deviceDataOut[index]= 127 - (32 - (deviceDataIn[index] - key));    
            } else {
                deviceDataOut[index]=deviceDataIn[index] - key;    
            }
        }
        else {
            deviceDataOut[index] = deviceDataIn[index];
        }
    }
}


int readData(string fileName, char *data) {

  streampos size;

  ifstream file (fileName, ios::in|ios::binary|ios::ate);
  if (file.is_open())
  {
    size = file.tellg();
    file.seekg (0, ios::beg);
    file.read (data, size);
    file.close();

    cout << "The entire file content is in memory." << endl;
  }
  else cout << "Unable to open file" << endl;
  return 0;
}

int writeData(int size, string fileName, char *data) {
  ofstream file (fileName, ios::out|ios::binary|ios::trunc);

  if (file.is_open())
  {
    file.write (data, size);
    file.close();

    cout << "The entire file content was written to " << fileName << "." << endl;
    return 0;
  }
  else cout << "Unable to open file";

  return -1; 
}

int EncryptSeq (int n, int largeKey[], int largeKeySize, char* data_in, char* data_out) 
{  
    int i, key, key_index;
    timer sequentialTime = timer("Sequential encryption");    
    sequentialTime.start();

    key_index = 0;
    key = largeKey[key_index % largeKeySize];
    for (i=0; i<n; i++) { 
        if (data_in[i] >= 32) {
            if (data_in[i] + key >= 127) {
                data_out[i] = ((data_in[i] + key) % 127) + 32;
            } else {
                data_out[i]= data_in[i] + key;
            }
            key_index++;
            key = largeKey[key_index % largeKeySize];
        } else {
            data_out[i] = data_in[i];
        }
    }
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Encryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return 0; 
}

int DecryptSeq(int n, int largeKey[], int largeKeySize, char* data_in, char* data_out)
{
    int i, key, key_index;
    timer sequentialTime = timer("Sequential decryption");

    key_index = 0;
    sequentialTime.start();
    key = largeKey[key_index % largeKeySize];
    for (i=0; i<n; i++) {
        if (data_in[i] >= 32) {
            if (data_in[i] - key < 32) {
                data_out[i]= 127 - (32 - (data_in[i] - key));    
            } else {
                data_out[i]=data_in[i] - key;    
            }
            key_index++;
            key = largeKey[key_index % largeKeySize];
        }
        else {
            data_out[i] = data_in[i];
        }
    }
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Decryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return 0;
}


int EncryptCuda(int n, int largeKey[], int largeKeySize, char* data_in, char* data_out) {
    int threadBlockSize = 1024;

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char ), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(largeKeyDevice, largeKey, largeKeySize*sizeof(int),cudaMemcpyHostToDevice));
    memoryTime.stop();
    // execute kernel
    kernelTime1.start();
    encryptKernel<<<(n/threadBlockSize)+1, threadBlockSize>>>(n, largeKeySize, deviceDataIn, deviceDataOut, largeKeyDevice);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char ), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    cout << fixed << setprecision(6);
    cout << "Encrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Encrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}

int DecryptCuda(int n,  int largeKey[], int largeKeySize, char* data_in, char* data_out) {
    int threadBlockSize = 1024;

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char ), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(largeKeyDevice, largeKey, largeKeySize*sizeof(int),cudaMemcpyHostToDevice));
    memoryTime.stop();
    
    // execute kernel
    kernelTime1.start();
    decryptKernel<<<(n/threadBlockSize)+1, threadBlockSize>>>(n, largeKeySize, deviceDataIn, deviceDataOut, largeKeyDevice);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char ), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    cout << fixed << setprecision(6);
    cout << "Decrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Decrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

   return 0;
}


int fileSize(string filename) {
  int size; 
  ifstream file (filename, ios::in|ios::binary|ios::ate);

  if (file.is_open())
  {
    size = file.tellg();
    file.close();
  }
  else {
    cout << "Unable to open file";
    size = -1; 
  }
  return size; 
}


void setup_CUDAmem(int n, int largeKeySize) {
    deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        exit(1);
    }
    deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        exit(1);
    }

    largeKeyDevice = NULL;
    checkCudaCall(cudaMalloc((void **) &largeKeyDevice, largeKeySize * sizeof(int)));
    if (largeKeyDevice == NULL) {
        checkCudaCall(cudaFree(largeKeyDevice));
        cout << "could not allocate memory!" << endl;
        exit(1);
    }
}

void teardown_CUDAmem() {
    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(largeKeyDevice));
}

int main(int argc, char* argv[]) {
    int n;
    int largeKeySize = 4;
    int largeKey[] = {1, 2, 3, 4};


    char* filename;
    if (argc == 0) {
        cout << "Usage: ./crypto {filename}" << endl;
    }
    filename = argv[1];
    n = fileSize(filename);
    if (n == -1) {
    	cout << "File not found! Exiting ... " << endl; 
    	exit(0);
    }
    
    char* data_in = (char*)malloc(sizeof(char)*n);
    char* data_out = (char*)malloc(sizeof(char)*n);
    char* data_in_cuda = (char*)malloc(sizeof(char)*n);
    char* data_out_cuda = (char*)malloc(sizeof(char)*n);  
     
    for (int i = 0; i < n; i++) {
        data_in[i] = ' ';
        data_out[i] = ' ';
        data_in_cuda[i] = ' ';
        data_out_cuda[i] = ' ';
    }

    readData(filename, data_in);

    cout << "Encrypting a file of " << n << " characters." << endl;

    string sequential_out_file = "sequential_largeKey.data";
    string cuda_out_file = "cuda_largeKey.data";

    string sequential_decrypted = "sequential_decrypted_largeKey.data";
    string cuda_decrypted = "recovered_largeKey.data";

    setup_CUDAmem(n, largeKeySize);

    EncryptSeq(n, largeKey, largeKeySize, data_in, data_out);

    writeData(n, sequential_out_file, data_out);

    EncryptCuda(n, largeKey, largeKeySize, data_in, data_out_cuda);
    writeData(n, cuda_out_file, data_out_cuda);  

    readData(cuda_out_file, data_in_cuda);
    readData(sequential_out_file, data_in);

    cout << "Decrypting a file of " << n << "characters" << endl;
    DecryptSeq(n, largeKey, largeKeySize, data_in, data_out);
    writeData(n, sequential_decrypted, data_out);
    DecryptCuda(n, largeKey, largeKeySize, data_in_cuda, data_out_cuda); 
    writeData(n, cuda_decrypted, data_out_cuda); 
 
    teardown_CUDAmem();

    delete[] data_in;
    delete[] data_out;
    delete[] data_in_cuda;
    delete[] data_out_cuda;

    return 0;
}
