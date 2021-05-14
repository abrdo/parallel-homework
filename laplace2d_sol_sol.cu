// compile:
// nvcc -arch=sm_60 -O3 laplace2d_sol.cu -o project1 -Xcompiler=-fopenmp


#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

__global__ void step_kernel(int imax, int jmax, const double* __restrict__ A,
                            double* Anew, double* reduction) {
  int tid_i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid_j = threadIdx.y + blockIdx.y * blockDim.y;
  // loop is on [1, imax] x [1, jmax]
  // starting index is (1,1)
  int i = tid_i + 1;
  int j = tid_j + 1;
  if (i < imax + 1 && j < jmax + 1) {
    int id_center = i + j * (imax + 2);
    int id_left = i - 1 + j * (imax + 2);
    int id_right = i + 1 + j * (imax + 2);
    int id_up = i + (j - 1) * (imax + 2);
    int id_down = i + (j + 1) * (imax + 2);
    Anew[id_center] =
        0.25f * (A[id_right] + A[id_left] + A[id_up] + A[id_down]);
    reduction[tid_i + tid_j * imax] = fabs(Anew[id_center] - A[id_center]);
  }
}

__global__ void copy_kernel(int imax, int jmax, double* A,
                            const double* __restrict__ Anew) {
  int tid_i = threadIdx.x + blockIdx.x * blockDim.x;
  int tid_j = threadIdx.y + blockIdx.y * blockDim.y;
  int i = tid_i + 1;
  int j = tid_j + 1;
  if (i < imax + 1 && j < jmax + 1) {
    int id = i + j * (imax + 2);
    A[id] = Anew[id];
  }
}


__global__ void max(double *d_max, double *d_data){
  extern __shared__ double maxs[]; // maximums in blocks
  extern __shared__ double temp[];
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  temp[tid] = d_data[ tid  +  bid * blockDim.y * blockDim.x];
  for (int d=(blockDim.y*blockDim.x)>>1; d>=1; d>>=1) {
    __syncthreads();
    if (tid<d)
      if (temp[tid] < temp[tid+d])
        temp[tid] = temp[tid+d];
    //if (tid==0)
    //  printf("%d %d %f  %f ", tid, d, temp[tid], temp[tid+d]);
  }
  
  if (tid == 0)  
    maxs[blockIdx.x + blockIdx.y * gridDim.x] = temp[0];
  

  if (tid == 0){
    for (int d=(gridDim.y*gridDim.x)>>1; d>=1; d>>=1){
      __threadfence();
      if (bid < d)
        if (temp[bid] < temp[bid+d])
          temp[bid] = temp[bid+d];
    }
  
    if (bid == 0)
      *d_max = maxs[0];
  }
  
}



int main(int argc, const char** argv) {
  // Size along y
  int jmax = 4096;
  // Size along x
  int imax = 4096;
  // Size along x
  int iter_max = 100;

  double pi = 2.0 * asin(1.0);
  const double tol = 1.0e-6;
  double error = 1.0;
  // we need imax*jmax thread to do so, choose an arbitrary block shape:
  dim3 nthreads(16, 8);
  // start enough blocks to get enough thread (rounding up)
  dim3 nblocks((imax - 1) / nthreads.x + 1, (jmax - 1) / nthreads.y + 1);

  double* A = new double[(imax + 2) * (jmax + 2)];
  double* Anew = new double[(imax + 2) * (jmax + 2)];
  memset(A, 0, (imax + 2) * (jmax + 2) * sizeof(double));

  // set boundary conditions
  for (int i = 0; i < imax + 2; i++) A[(0) * (imax + 2) + i] = 0.0;

  for (int i = 0; i < imax + 2; i++) A[(jmax + 1) * (imax + 2) + i] = 0.0;

  for (int j = 0; j < jmax + 2; j++) {
    A[(j) * (imax + 2) + 0] = sin(pi * j / (jmax + 1));
  }

  for (int j = 0; j < imax + 2; j++) {
    A[(j) * (imax + 2) + imax + 1] = sin(pi * j / (jmax + 1)) * exp(-pi);
  }

  printf("Jacobi relaxation Calculation: %d x %d mesh\n", imax + 2, jmax + 2);

  int iter = 0;

  for (int i = 1; i < imax + 2; i++) Anew[(0) * (imax + 2) + i] = 0.0;

  for (int i = 1; i < imax + 2; i++) Anew[(jmax + 1) * (imax + 2) + i] = 0.0;

  for (int j = 1; j < jmax + 2; j++)
    Anew[(j) * (imax + 2) + 0] = sin(pi * j / (jmax + 1));

  for (int j = 1; j < jmax + 2; j++)
    Anew[(j) * (imax + 2) + jmax + 1] = sin(pi * j / (jmax + 1)) * expf(-pi);

  // Allocate and move device memory
  double *A_d, *Anew_d, *error_d, *max_error_d;
  cudaMalloc((void**)&A_d, (imax + 2) * (jmax + 2) * sizeof(double));
  cudaMalloc((void**)&Anew_d, (imax + 2) * (jmax + 2) * sizeof(double));
  cudaMalloc((void**)&error_d, imax * jmax * sizeof(double));
  cudaMalloc((void**)&max_error_d, nblocks.x * nblocks.y * sizeof(double));
  cudaMemcpy(A_d, A, (imax + 2) * (jmax + 2) * sizeof(double),
             cudaMemcpyHostToDevice);
  cudaMemcpy(Anew_d, Anew, (imax + 2) * (jmax + 2) * sizeof(double),
             cudaMemcpyHostToDevice);

  while (error > tol && iter < iter_max) {
    error = 0.0;

    // loops on [1, imax]x[1,jmax]

    step_kernel<<<nblocks, nthreads>>>(imax, jmax, A_d, Anew_d, error_d);
    
    /*
    error = thrust::reduce(thrust::device_ptr<double>(error_d),
                           thrust::device_ptr<double>(error_d + imax * jmax),
                           0.0, thrust::maximum<double>());
    
    */
    max<<<nblocks, nthreads, nthreads.x*nthreads.y*sizeof(double)>>>(max_error_d, error_d);
    cudaMemcpy(&error, max_error_d, sizeof(double), cudaMemcpyDeviceToHost);

    

    copy_kernel<<<nblocks, nthreads>>>(imax, jmax, A_d, Anew_d);

    if (iter % 10 == 0) printf("%5d, %0.6f\n", iter, error);
    iter++;
  }


  printf("%5d, %0.6f\n", iter, error);

  double err_diff = fabs((100.0 * (error / 2.421354960840227e-03)) - 100.0);
  printf("Total error is within %3.15E %% of the expected error\n", err_diff);
  if (err_diff < 0.001)
    printf("This run is considered PASSED\n");
  else
    printf("This test is considered FAILED\n");

  cudaFree(A_d);
  cudaFree(Anew_d);
  cudaFree(error_d);
  cudaFree(max_error_d);
  delete[] A;
  delete[] Anew;

  return 0;
}
