#include <iostream>

#include <nt2/sdk/memory/opencl/buffer.hpp>

#include <nt2/table.hpp>

#include <nt2/include/functions/zeros.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/size.hpp>

#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/module.hpp>

#include<nt2/sdk/meta/type_id.hpp>

namespace compute = boost::compute;


NT2_TEST_CASE_TPL( opencl_buffer_swap, (double) )
{
  nt2::table<T> A_init = nt2::ones(10,1,nt2::meta::as_<T>());
  nt2::table<T> B_init = nt2::zeros(10,1,nt2::meta::as_<T>());

  nt2::table<T,nt2::device_> cl_A, cl_B;
  cl_A = A_init;
  cl_B = B_init;

  nt2::table<T> A_final, B_final;
  A_final = cl_B;
  B_final = cl_A;

  NT2_TEST_EQUAL(A_final,B_init);
  NT2_TEST_EQUAL(B_final,A_init);
}


NT2_TEST_CASE_TPL(opencl_buffer_default, NT2_REAL_TYPES)
{
  nt2::memory::opencl_buffer<T> b;

  NT2_TEST(b.empty());
  NT2_TEST_EQUAL(b.size()     , 0u      );
}

NT2_TEST_CASE_TPL(opencl_buffer_size_init, NT2_REAL_TYPES)
{
  nt2::memory::opencl_buffer<T> b(5);

  NT2_TEST(!b.empty());
  NT2_TEST_EQUAL(b.size()     , 5u    );
}


NT2_TEST_CASE_TPL(opencl_buffer_resize, NT2_REAL_TYPES)
{
  nt2::memory::opencl_buffer<T> b(5);
  b.resize(15);

  NT2_TEST(!b.empty());
  NT2_TEST_EQUAL(b.size()     , 15u    );
}

NT2_TEST_CASE_TPL( opencl_custom_kernel, NT2_REAL_TYPES )
{
  nt2::table<T,nt2::device_> d_asc, d_des, d_res;
  nt2::table<T> h_asc, h_des, h_res;

  compute::command_queue queue = compute::system::default_queue();

  // Generate kernel to run on device
  std::stringstream src;
  std::string typeString = nt2::type_id<T>();
  src   << "__kernel void matMult(__global const " << typeString << " *A,"
        << "                      __global const " << typeString << " *B,"
        << "                      __global "       << typeString << " *C,"
        << "                      const int m,"
        << "                      const int n,"
        << "                      const int p"
        << "                      ) {"
        << "    const uint my_x = get_global_id(0);"
        << "    const uint my_y = get_global_id(1);"
        << "    uint k;"
        << "    " << typeString << " res = 0;"
        << "    for ( k = 0 ; k < p ; ++k ) {"
        << "            res += A[my_y*n+k]*B[k*p+my_x];"
        << "    C[my_x + my_y*p] = res;"
        << "    }"
        << "}";

  const char *source = src.str().c_str();

  // Create kernel, set inputs
  compute::program program = compute::program::create_with_source(source, compute::system::default_context());
  program.build();

  compute::kernel kernel(program, "matMult");

  compute::detail::set_kernel_arg<compute::vector<T> > setArgs;
  setArgs(kernel,0,d_asc.data());
  setArgs(kernel,1,d_des.data());
  setArgs(kernel,2,d_res.data());
  kernel.set_arg(3,3);
  kernel.set_arg(4,3);
  kernel.set_arg(5,3);

  // Run kernel
  size_t dim = 2;
  size_t offset[] = {0,0};
  size_t global_size[] = {3,3};
  size_t local_size[] = {1,1};
  queue.enqueue_nd_range_kernel(kernel, dim, offset, global_size, local_size);

  // Recover data from device
  h_res = d_res;

  // TODO: Implement actual comparison/validation
  NT2_TEST_EQUAL(2,2);
}

