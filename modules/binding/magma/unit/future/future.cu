#include "Obj_cuda.hpp"
#include <stdio.h>

__global__ void compute(int* res, int* a, int* b)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
res[i] = a[i] + b[i];
}


void Obj_cuda::call(int* res, int* a, int* b ,int N)
{
compute<<<1,1>>>(res,a,b);
}

