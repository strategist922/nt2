#include <nt2/arithmetic/functions/opencl/divides.hpp>
#include <nt2/arithmetic/functions/opencl/plus.hpp>
#include <CL/cl.h>
#include <boost/compute/container/vector.hpp>
#include <string>

namespace compute = boost::compute;


std::string plus4 ()
{
  std::string res("");
  res += std::string("inline float divides( float arg0, float arg1 )");
  res += nt2::opencl::divides() + std::string("\n");
  res += std::string("inline float plus( float arg0, float arg1 )");
  res += nt2::opencl::plus() + std::string("\n");
  res += std::string("__kernel void plus4 ( float* t0, const float*  t1, const float*  t2, const float*  t3)\n{\n");
  res += std::string("  int index = get_global_id(0)\n");
  res += std::string("  t0[index] = plus(log(divides(t1[index],t2[index])),t3[index]);\n");
  res += std::string("}\n");

  return res;
}
void plus4_wrapper( compute::vector< float > & t0, const compute::vector< float > &  t1, const compute::vector< float > &  t2, const compute::vector< float > &  t3, std::size_t dimGrid, std::size_t blockDim, std::size_t gridNum, std::size_t blockNum, compute::command_queue & queue)
{
  compute::program program = 
    compute::program::create_with_source(plus4(), queue.get_context());
  program.build();

  compute::kernel kernel(program, "plus4");
  kernel.set_arg(0 , t0);
  kernel.set_arg(1 , t1);
  kernel.set_arg(2 , t2);
  kernel.set_arg(3 , t3);

  size_t dim = 1;
  size_t offset[] = { (dimGrid * gridNum) + (blockDim * blockNum) };
  size_t global_size[] = { dimGrid };
  size_t local_size[] = { blockDim };
  queue.enqueue_nd_range_kernel(kernel, dim, offset, global_size, local_size);

}
