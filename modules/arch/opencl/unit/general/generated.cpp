#include <nt2/sdk/external_kernel/external_kernel.hpp>
#include <nt2/sdk/opencl/opencl.hpp>
#include <nt2/table.hpp>
#include <nt2/core/functions/transform.hpp>
#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>
#include <nt2/include/functions/numel.hpp>
#include <nt2/include/functions/log.hpp>
#include <CL/cl.h>
#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

#include "generated_cl.cpp"

using boost::proto::child_c;
namespace compute = boost::compute;

//void plus4_wrapper( compute::vector< float > & t0, const compute::vector< float > &  t1, const compute::vector< float > &  t2, const compute::vector< float > &  t3, std::size_t dimGrid, std::size_t blockDim, std::size_t gridNum, std::size_t blockNum, compute::command_queue & queue);


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
  compute::command_queue queues[nQueues];



   //------------ transfers host to device ------------//

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

      //------------------ kernel call -------------------//

  std::size_t spill;// = ( leftover != 0 ) ? ( 1 : 0 );
  if ( leftover != 0 ) spill = 1;
  else spill = 0;

  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {
    std::size_t j = i % nQueues;

//    plus4_wrapper(
//      boost::proto::value(a0).data(),
//      boost::proto::value(child_c<0>(child_c<0>(child_c<0>(a1)))).data(),
//      boost::proto::value(child_c<1>(child_c<0>(child_c<0>(a1)))).data(),
//      boost::proto::value(child_c<1>(a1)).data(),
//      dimGrid, blockDim, 0, i, queues[j]
//    );
  }// for i in n

      //------------ transfers device to host ------------//

} // kernel
} // namespace nt2

