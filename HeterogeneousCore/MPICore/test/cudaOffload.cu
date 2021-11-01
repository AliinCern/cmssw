
#include <cuda.h>
#include "cudaOffload.h"

//////////////////////////////////////////// C U D A  /////////////////////////////////////////
//called in the Host and excuted in the Device (GPU)
__global__ void addVectorsGpuNonStrided(float *vect1, float *vect2, float *vect3) //add two vectors and save the result into the third vector.
{
  //blockDim.x gives the number of threads in a block, in the x direction.
  //gridDim.x gives the number of blocks in a grid, in the x direction.
  //blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction, in this case).
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  vect3[i] = vect2[i] + vect1[i];
}

//called in the Host and excuted in the Device (GPU)
__global__ void addVectorsGpu(float *vect1, float *vect2, float *vect3, int size) //add two vectors and save the result into the third vector.
{
  //blockDim.x gives the number of threads in a block, in the x direction.
  //gridDim.x gives the number of blocks in a grid, in the x direction.
  //blockDim.x * gridDim.x gives the number of threads in a grid (in the x direction, in this case).
  int first = blockDim.x*blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = first; i < size; i+= stride)
  {
    vect3[i] = vect2[i] + vect1[i];
  }
}

void cudaVectorAdd(std::vector<float>& vectorWorkers1, const std::vector<float>& vectorWorkers2){
    // for (long unsigned int i = 0; i < mpiInput.vectorWorkers1.size(); i++) {
    //   mpiInput.vectorWorkers1[i] += mpiInput.vectorWorkers2[i];
    // }

    float *d_vect1, *d_vect2, *d_vect3Gpu; //create pointers for Device.

    cudaMalloc((void**)&d_vect1, vectorWorkers1.size());//allocate memory space for vector in the global memory of the Device.
    cudaMalloc((void**)&d_vect2, vectorWorkers2.size());
    cudaMalloc((void**)&d_vect3Gpu, vectorWorkers1.size());

    cudaMemcpy(d_vect1, vectorWorkers1.data(), vectorWorkers1.size() * sizeof (float), cudaMemcpyHostToDevice);//copy random vector from host to device.
    cudaMemcpy(d_vect2, vectorWorkers2.data(), vectorWorkers1.size() * sizeof (float), cudaMemcpyHostToDevice);

    int threads = 512; //arbitrary number.
    int blocks = (vectorWorkers1.size() + threads - 1) / threads; //get ceiling number of blocks.
    //blocks = std::min(blocks, 8); // Number 8 is least number can be got from lowest Nevedia GPUs.
    
    addVectorsGpuNonStrided<<<blocks,threads>>> (d_vect1, d_vect2, d_vect3Gpu); //call device function to add two vectors and save into vect3Gpu.
    
    cudaMemcpy(vectorWorkers1.data(), d_vect3Gpu, vectorWorkers1.size() * sizeof (float), cudaMemcpyHostToDevice);
}

