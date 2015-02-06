#include "Obj_cuda.hpp"

__global__ void compute(int* res, int* a, int* b)
{
int i = blockIdx.x*blockDim.x + threadIdx.x;
res[i] = 10 *(a[i] + b[i]);
}

int Obj_cuda::operator()(int* res, int* a, int* b ,int N = 10)
{
compute<<<1,N>>>(res,a,b);

return 1;
}
