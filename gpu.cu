#include "gpu.hpp"
#include <thrust/sort.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void cu_grey(uint32_t *img, int w, int h) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

/*  int x = idx % w;
  int y = idx / w;

  if (x >= w)
    return;
  if (y >= h)
    return;*/

  if (idx > w * h)
    return;

  printf("idx: %d\n", idx);

  rgbx8888_u pix ={ .value = img[idx]};

  auto r = pix.argb.r;
  auto g = pix.argb.g;
  auto b = pix.argb.b;

  double grey_val = 0.3*(double)r + 0.59*(double)g + 0.11*(double)b;
  pix.argb.r = pix.argb.g = pix.argb.b = (unsigned char)((int)grey_val % 256);
  img[idx] = pix.value;
}

void gpu_grey(uint32_t *img, int w, int h) {
  size_t threadsByGrid = 1024;
  size_t gridNumber = ((w * h) / threadsByGrid) + 1;
  uint32_t *d_img = nullptr;

  std::cout
    << "threads: " << threadsByGrid << " "
    << "grids:   " << gridNumber << std::endl;
  cudaMalloc(&d_img, w * h * sizeof(uint32_t));
  gpuErrchk(cudaPeekAtLastError());
  cudaMemcpy(d_img, img, w * h * sizeof(uint32_t), cudaMemcpyHostToDevice);
  gpuErrchk(cudaPeekAtLastError());

  cu_grey<<<gridNumber, threadsByGrid>>>(d_img, w, h);
  gpuErrchk(cudaPeekAtLastError());

  cudaMemcpy(img, d_img, w * h * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  gpuErrchk(cudaPeekAtLastError());
  cudaFree(d_img);
  gpuErrchk(cudaPeekAtLastError());
}
