#include "Obj_cuda.hpp"


__global__ void compute(int* res, int a)
{
res[0] = a+30;
}


void Obj_cuda::call(int* res, int a,N)
{
compute<<<1,1>>>(res,a);
}

