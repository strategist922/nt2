#include <nt2/arithmetic/functions/opencl/divides.hpp>
#include <nt2/include/functions/log.hpp>
#include <nt2/arithmetic/functions/opencl/plus.hpp>
#include <CL/cl.h>
#include <boost/compute/container/vector.hpp>
#include <string>

#include <nt2/sdk/meta/type_id.hpp>
#include <iostream>

namespace compute = boost::compute;


std::string plus4 ()
{
  std::string res("");
  res += std::string("inline float divides( float arg0, float arg1 )");
  res += nt2::opencl::divides() + std::string("\n");
  res += std::string("inline float plus( float arg0, float arg1 )");
  res += nt2::opencl::plus() + std::string("\n");
  res += std::string("__kernel void plus4 ( __global float* t0, __global const float*  t1, __global const float*  t2, __global const float*  t3)\n{\n");
  res += std::string("  int index = get_global_id(0);\n");
  res += std::string("  t0[index] = plus(log(divides(t1[index],t2[index])),t3[index]);\n");
  res += std::string("}\n");

  return res;
}
void plus4_wrapper( compute::vector< float > & t0, const compute::vector< float > &  t1, const compute::vector< float > &  t2, const compute::vector< float > &  t3, std::size_t dimGrid, std::size_t blockDim, std::size_t gridNum, std::size_t blockNum, compute::command_queue & queue)
{
  std::string code = plus4();
  compute::program program =
    compute::program::create_with_source(code.c_str(), queue.get_context());
  program.build();

  std::cout << nt2::type_id(t0) << std::endl;
  std::cout << t0.size() << "\t"
            << t1.size() << "\t"
            << t2.size() << "\t"
            << t3.size() << "\t"
            << "\n";

//  std::cout << plus4() << "\n";

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


//-----------------------------------------------

#include <nt2/sdk/external_kernel/external_kernel.hpp>
#include <nt2/sdk/opencl/opencl.hpp>
#include <nt2/core/functions/transform.hpp>
#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>
#include <nt2/include/functions/numel.hpp>
#include <CL/cl.h>
#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <nt2/sdk/opencl/settings/specific_data.hpp>

using boost::proto::child_c;
namespace compute = boost::compute;


namespace nt2 {

template<> template <>
void nt2::external_kernel<nt2::tag::transform_, nt2::tag::opencl_<boost::simd::tag::avx2_> >::call<nt2::container::view<nt2::tag::table_, float, nt2::settings ()> const, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::plus_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::log_, boost::proto::argsns_::list1<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::divides_, boost::proto::argsns_::list2<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > const> (nt2::container::view<nt2::tag::table_, float, nt2::settings ()> const& a0, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::plus_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::log_, boost::proto::argsns_::list1<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::divides_, boost::proto::argsns_::list2<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > const& a1)
{
  std::size_t size = numel(boost::proto::child_c<0>(a1));
  std::size_t blockSize = std::min(std::size_t(256),size);
  std::size_t nQueues = std::min(std::size_t(4),size/blockSize);
  std::size_t leftover = size % blockSize ;
  std::size_t n = size / blockSize;
//  std::size_t n = (size + blockSize - 1) / blockSize;
  std::size_t dimGrid  = blockSize;
  std::size_t blockDim = 1;
  compute::command_queue *queues = new compute::command_queue[nQueues];

  std::size_t spill;
  if ( leftover != 0 ) spill = 1;
  else spill = 0;

  std::vector<compute::device> devices;
  for ( compute::device dev : compute::system::devices() )
    devices.push_back(dev);

// TODO: take advantage of multiple devices
  std::vector<compute::context> contexts;
  contexts.resize(devices.size());
  for ( std::size_t i = 0 ; i < devices.size() ; ++i )
    contexts[i] = compute::context(devices[i]);
  for ( std::size_t i = 0 ; i < nQueues ; ++i )
    queues[i] = compute::command_queue(contexts[0], devices[0]);
for ( std::size_t i = 0 ; i < nQueues ; ++i )
std::cout << queues[i].get_context() << "\n";

  std::cout << "ALLOCATING\n\n";

//------------ allocate necessary device memory ------------//
  boost::proto::value(a0).specifics().allocate(blockSize,nQueues,size, queues);
  boost::proto::value(child_c<0>(child_c<0>(child_c<0>(a1)))).specifics().allocate(blockSize,nQueues,size, queues);
  boost::proto::value(child_c<1>(child_c<0>(child_c<0>(a1)))).specifics().allocate(blockSize,nQueues,size, queues);
  boost::proto::value(child_c<1>(a1)).specifics().allocate(blockSize,nQueues,size, queues);

  std::cout << "TRANSFER H2D\n\n";

   //------------ transfers host to device ------------//
  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {
    std::size_t j = i % nQueues;
    std::size_t extra = 0;
    if ( i == n ) extra = leftover;
    boost::proto::value(a0).specifics().transfer_htd(a0, i, queues[j], j );
    boost::proto::value(child_c<0>(child_c<0>(child_c<0>(a1)))).specifics().transfer_htd(child_c<0>(child_c<0>(child_c<0>(a1))), i, queues[j], j );
    boost::proto::value(child_c<1>(child_c<0>(child_c<0>(a1)))).specifics().transfer_htd(child_c<1>(child_c<0>(child_c<0>(a1))), i, queues[j], j );
    boost::proto::value(child_c<1>(a1)).specifics().transfer_htd(child_c<1>(a1), i, queues[j], j );
  }


      //------------------ kernel call -------------------//

 std::cout << "Launching kernel\n";

  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {
    std::size_t j = i % nQueues;
    plus4_wrapper(
      boost::proto::value(a0).specifics().data(j),
      boost::proto::value(child_c<0>(child_c<0>(child_c<0>(a1)))).specifics().data(j),
      boost::proto::value(child_c<1>(child_c<0>(child_c<0>(a1)))).specifics().data(j),
      boost::proto::value(child_c<1>(a1)).specifics().data(j),
      dimGrid, blockDim, 0, i, queues[j]
    );
  }// for i in n

  std::cout << "Kernel Complete\n";

      //------------ transfers device to host ------------//

  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {
    std::size_t j = i % nQueues;
    std::size_t extra = 0;
    if ( i == n ) extra = leftover;
    boost::proto::value(a0).specifics().transfer_dth(a0, n, queues[j], j, leftover );
}

  std::cout << "Dev to host complete\n";


delete [] queues;


} // kernel
} // namespace nt2
