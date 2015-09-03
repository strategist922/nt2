#include <nt2/table.hpp>
#include <nt2/arithmetic/functions/opencl/divides.hpp>
#include <nt2/include/functions/divides.hpp>
#include <nt2/include/functions/exp.hpp>
#include <nt2/arithmetic/functions/opencl/fastnormcdf.hpp>
#include <nt2/include/functions/fastnormcdf.hpp>
#include <nt2/arithmetic/functions/opencl/fnms.hpp>
#include <nt2/include/functions/fnms.hpp>
#include <nt2/include/functions/log.hpp>
#include <nt2/arithmetic/functions/opencl/multiplies.hpp>
#include <nt2/include/functions/multiplies.hpp>
#include <nt2/arithmetic/functions/opencl/plus.hpp>
#include <nt2/include/functions/plus.hpp>
#include <nt2/include/functions/sqrt.hpp>
#include <CL/cl.h>
#include <boost/compute/container/vector.hpp>
#include <string>

namespace compute = boost::compute;


std::string tie20 ()
{
  std::string res("");
  res += std::string("inline float divides( float arg0, float arg1 )");
  res += nt2::opencl::divides() + std::string("\n");
  res += std::string("inline float multiplies( float arg0, float arg1 )");
  res += nt2::opencl::multiplies() + std::string("\n");
  res += std::string("inline float plus( float arg0, float arg1 )");
  res += nt2::opencl::plus() + std::string("\n");
  res += std::string("inline float fnms( float arg0, float arg1, float arg2 )");
  res += nt2::opencl::fnms() + std::string("\n");
  res += std::string("inline float fastnormcdf( float arg0 )");
  res += nt2::opencl::fastnormcdf() + std::string("\n");
  res += std::string("__kernel void tie20 ( __global  float*  t0 , __global  float*  t1 , __global  float*  t2 , __global  float*  t3,  __global const float*   t4,  __global const float*   t5,  __global const float*   t6, const float t7,  __global const float*   t8, const float t9,  __global const float*   t10, const float t11,  __global const float*   t12,  __global const float*   t13,  __global const float*   t14, const float t15,  __global const float*   t16,  __global const float*   t17,  __global const float*   t18,  __global const float*   t19)\n{\n");
  res += std::string("  int index = get_global_id(0);\n");
  res += std::string("   t0[index] = sqrt(t4[index]); t1[index] = plus(log(divides(t5[index],t6[index])),divides(multiplies(t7,t8[index]),multiplies(t9,t10[index]))); t2[index] = fnms(t11,t12[index],t13[index]); t3[index] = fnms(multiplies(t14[index],exp(multiplies(t15,t16[index]))),fastnormcdf(t17[index]),multiplies(t18[index],fastnormcdf(t19[index])));\n");
  res += std::string("}\n");

  return res;
}
void tie20_wrapper( compute::vector< float > & t0 , compute::vector< float > & t1 , compute::vector< float > & t2 , compute::vector< float > & t3, const compute::vector< float > &  t4, const compute::vector< float > &  t5, const compute::vector< float > &  t6, const float t7, const compute::vector< float > &  t8, const float t9, const compute::vector< float > &  t10, const float t11, const compute::vector< float > &  t12, const compute::vector< float > &  t13, const compute::vector< float > &  t14, const float t15, const compute::vector< float > &  t16, const compute::vector< float > &  t17, const compute::vector< float > &  t18, const compute::vector< float > &  t19, std::size_t dimGrid, std::size_t blockDim, std::size_t gridNum, std::size_t blockNum, compute::command_queue & queue)
{
  compute::program program = 
    compute::program::create_with_source(tie20().c_str(), queue.get_context());
  program.build();

  compute::kernel kernel(program, "tie20");
  kernel.set_arg(0 , t0);
  kernel.set_arg(1 , t1);
  kernel.set_arg(2 , t2);
  kernel.set_arg(3 , t3);
  kernel.set_arg(4 , t4);
  kernel.set_arg(5 , t5);
  kernel.set_arg(6 , t6);
  kernel.set_arg(7 , t7);
  kernel.set_arg(8 , t8);
  kernel.set_arg(9 , t9);
  kernel.set_arg(10 , t10);
  kernel.set_arg(11 , t11);
  kernel.set_arg(12 , t12);
  kernel.set_arg(13 , t13);
  kernel.set_arg(14 , t14);
  kernel.set_arg(15 , t15);
  kernel.set_arg(16 , t16);
  kernel.set_arg(17 , t17);
  kernel.set_arg(18 , t18);
  kernel.set_arg(19 , t19);

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
#include <nt2/include/functions/tie.hpp>

using boost::proto::child_c;
namespace compute = boost::compute;

void tie20_wrapper( compute::vector< float > & t0 , compute::vector< float > & t1 , compute::vector< float > & t2 , compute::vector< float > & t3, const compute::vector< float > &  t4, const compute::vector< float > &  t5, const compute::vector< float > &  t6, const float t7, const compute::vector< float > &  t8, const float t9, const compute::vector< float > &  t10, const float t11, const compute::vector< float > &  t12, const compute::vector< float > &  t13, const compute::vector< float > &  t14, const float t15, const compute::vector< float > &  t16, const compute::vector< float > &  t17, const compute::vector< float > &  t18, const compute::vector< float > &  t19, std::size_t dimGrid, std::size_t blockDim, std::size_t gridNum, std::size_t blockNum, compute::command_queue & queue);


namespace nt2 {

template<> template <>
void nt2::external_kernel<nt2::tag::transform_, nt2::tag::opencl_<boost::simd::tag::avx2_> >::call<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::tie_, boost::proto::argsns_::list4<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, 0l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, 0l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, 0l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, 0l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&> >, 4l>, boost::fusion::vector4<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&> > const, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::tie_, boost::proto::argsns_::list4<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::sqrt_, boost::proto::argsns_::list1<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::plus_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::log_, boost::proto::argsns_::list1<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::divides_, boost::proto::argsns_::list2<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::divides_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<float>, 0l>, float>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<float>, 0l>, float>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::fnms_, boost::proto::argsns_::list3<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<float>, 0l>, float>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 3l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::fnms_, boost::proto::argsns_::list3<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::exp_, boost::proto::argsns_::list1<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<float>, 0l>, float>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::fastnormcdf_, boost::proto::argsns_::list1<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::fastnormcdf_, boost::proto::argsns_::list1<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 3l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 4l>, boost::fusion::vector4<nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > > const> (nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::tie_, boost::proto::argsns_::list4<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, 0l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, 0l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, 0l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&>, 0l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&> >, 4l>, boost::fusion::vector4<nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&, nt2::memory::container<nt2::tag::table_, float, nt2::settings ()>&> > const& a0, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::tie_, boost::proto::argsns_::list4<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::sqrt_, boost::proto::argsns_::list1<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::plus_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::log_, boost::proto::argsns_::list1<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::divides_, boost::proto::argsns_::list2<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::divides_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<float>, 0l>, float>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<float>, 0l>, float>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::fnms_, boost::proto::argsns_::list3<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<float>, 0l>, float>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 3l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::fnms_, boost::proto::argsns_::list3<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::exp_, boost::proto::argsns_::list1<nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::terminal_, boost::proto::argsns_::term<float>, 0l>, float>, nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::fastnormcdf_, boost::proto::argsns_::list1<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> >, nt2::container::expression<boost::proto::exprns_::basic_expr<boost::simd::tag::multiplies_, boost::proto::argsns_::list2<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()>, nt2::container::expression<boost::proto::exprns_::basic_expr<nt2::tag::fastnormcdf_, boost::proto::argsns_::list1<nt2::container::view<nt2::tag::table_, float const, nt2::settings ()> >, 1l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 2l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 3l>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > >, 4l>, boost::fusion::vector4<nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)>, nt2::memory::container<nt2::tag::table_, float, nt2::settings (nt2::of_size_<-1l, -1l, -1l, -1l>)> > > const& a1)
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
    queues[i] = compute::command_queue(compute::system::default_context(), compute::system::default_device());

//------------ allocate necessary device memory ------------//
  boost::proto::value(boost::proto::child_c<0>(a0)).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(boost::proto::child_c<1>(a0)).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(boost::proto::child_c<2>(a0)).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(boost::proto::child_c<3>(a0)).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<0>(child_c<0>(a1))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<0>(child_c<0>(child_c<0>(child_c<1>(a1))))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<1>(child_c<0>(child_c<0>(child_c<1>(a1))))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<1>(child_c<0>(child_c<1>(child_c<1>(a1))))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<1>(child_c<1>(child_c<1>(child_c<1>(a1))))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<1>(child_c<2>(a1))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<2>(child_c<2>(a1))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<0>(child_c<0>(child_c<3>(a1)))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<1>(child_c<0>(child_c<1>(child_c<0>(child_c<3>(a1)))))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<0>(child_c<1>(child_c<3>(a1)))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<0>(child_c<2>(child_c<3>(a1)))).specifics().allocate(blockSize,nQueues,size,queues);
  boost::proto::value(child_c<0>(child_c<1>(child_c<2>(child_c<3>(a1))))).specifics().allocate(blockSize,nQueues,size,queues);

   //------------ transfers host to device ------------//
  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {
    std::size_t j = i % nQueues;
    std::size_t extra = 0;
    if ( i == n ) extra = leftover;
    boost::proto::value(boost::proto::child_c<0>(a0)).specifics().transfer_htd(boost::proto::child_c<0>(a0), i, queues[j] , j );
    boost::proto::value(boost::proto::child_c<1>(a0)).specifics().transfer_htd(boost::proto::child_c<1>(a0), i, queues[j] , j );
    boost::proto::value(boost::proto::child_c<2>(a0)).specifics().transfer_htd(boost::proto::child_c<2>(a0), i, queues[j] , j );
    boost::proto::value(boost::proto::child_c<3>(a0)).specifics().transfer_htd(boost::proto::child_c<3>(a0), i, queues[j] , j );
    boost::proto::value(child_c<0>(child_c<0>(a1))).specifics().transfer_htd(child_c<0>(child_c<0>(a1)), i, queues[j], j );
    boost::proto::value(child_c<0>(child_c<0>(child_c<0>(child_c<1>(a1))))).specifics().transfer_htd(child_c<0>(child_c<0>(child_c<0>(child_c<1>(a1)))), i, queues[j], j );
    boost::proto::value(child_c<1>(child_c<0>(child_c<0>(child_c<1>(a1))))).specifics().transfer_htd(child_c<1>(child_c<0>(child_c<0>(child_c<1>(a1)))), i, queues[j], j );
    boost::proto::value(child_c<1>(child_c<0>(child_c<1>(child_c<1>(a1))))).specifics().transfer_htd(child_c<1>(child_c<0>(child_c<1>(child_c<1>(a1)))), i, queues[j], j );
    boost::proto::value(child_c<1>(child_c<1>(child_c<1>(child_c<1>(a1))))).specifics().transfer_htd(child_c<1>(child_c<1>(child_c<1>(child_c<1>(a1)))), i, queues[j], j );
    boost::proto::value(child_c<1>(child_c<2>(a1))).specifics().transfer_htd(child_c<1>(child_c<2>(a1)), i, queues[j], j );
    boost::proto::value(child_c<2>(child_c<2>(a1))).specifics().transfer_htd(child_c<2>(child_c<2>(a1)), i, queues[j], j );
    boost::proto::value(child_c<0>(child_c<0>(child_c<3>(a1)))).specifics().transfer_htd(child_c<0>(child_c<0>(child_c<3>(a1))), i, queues[j], j );
    boost::proto::value(child_c<1>(child_c<0>(child_c<1>(child_c<0>(child_c<3>(a1)))))).specifics().transfer_htd(child_c<1>(child_c<0>(child_c<1>(child_c<0>(child_c<3>(a1))))), i, queues[j], j );
    boost::proto::value(child_c<0>(child_c<1>(child_c<3>(a1)))).specifics().transfer_htd(child_c<0>(child_c<1>(child_c<3>(a1))), i, queues[j], j );
    boost::proto::value(child_c<0>(child_c<2>(child_c<3>(a1)))).specifics().transfer_htd(child_c<0>(child_c<2>(child_c<3>(a1))), i, queues[j], j );
    boost::proto::value(child_c<0>(child_c<1>(child_c<2>(child_c<3>(a1))))).specifics().transfer_htd(child_c<0>(child_c<1>(child_c<2>(child_c<3>(a1)))), i, queues[j], j );
  }


      //------------------ kernel call -------------------//

  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {
    std::size_t j = i % nQueues;
    tie20_wrapper(
      boost::proto::value(boost::proto::child_c<0>(a0)).specifics().data(j),
      boost::proto::value(boost::proto::child_c<1>(a0)).specifics().data(j),
      boost::proto::value(boost::proto::child_c<2>(a0)).specifics().data(j),
      boost::proto::value(boost::proto::child_c<3>(a0)).specifics().data(j),
      boost::proto::value(child_c<0>(child_c<0>(a1))).specifics().data(j),
      boost::proto::value(child_c<0>(child_c<0>(child_c<0>(child_c<1>(a1))))).specifics().data(j),
      boost::proto::value(child_c<1>(child_c<0>(child_c<0>(child_c<1>(a1))))).specifics().data(j),
      child_c<0>(child_c<0>(child_c<1>(child_c<1>(a1)))),
      boost::proto::value(child_c<1>(child_c<0>(child_c<1>(child_c<1>(a1))))).specifics().data(j),
      child_c<0>(child_c<1>(child_c<1>(child_c<1>(a1)))),
      boost::proto::value(child_c<1>(child_c<1>(child_c<1>(child_c<1>(a1))))).specifics().data(j),
      child_c<0>(child_c<2>(a1)),
      boost::proto::value(child_c<1>(child_c<2>(a1))).specifics().data(j),
      boost::proto::value(child_c<2>(child_c<2>(a1))).specifics().data(j),
      boost::proto::value(child_c<0>(child_c<0>(child_c<3>(a1)))).specifics().data(j),
      child_c<0>(child_c<0>(child_c<1>(child_c<0>(child_c<3>(a1))))),
      boost::proto::value(child_c<1>(child_c<0>(child_c<1>(child_c<0>(child_c<3>(a1)))))).specifics().data(j),
      boost::proto::value(child_c<0>(child_c<1>(child_c<3>(a1)))).specifics().data(j),
      boost::proto::value(child_c<0>(child_c<2>(child_c<3>(a1)))).specifics().data(j),
      boost::proto::value(child_c<0>(child_c<1>(child_c<2>(child_c<3>(a1))))).specifics().data(j),
      dimGrid, blockDim, 0, i, queues[j]
    );
  }// for i in n

      //------------ transfers device to host ------------//

  for ( std::size_t i = 0 ; i < n + spill ; ++i ) {
    std::size_t j = i % nQueues;
    std::size_t extra = 0;
    if ( i == n ) extra = leftover;
    boost::proto::value(boost::proto::child_c<0>(a0)).specifics().transfer_dth(boost::proto::child_c<0>(a0), n, queues[j], j , leftover );
    boost::proto::value(boost::proto::child_c<1>(a0)).specifics().transfer_dth(boost::proto::child_c<1>(a0), n, queues[j], j , leftover );
    boost::proto::value(boost::proto::child_c<2>(a0)).specifics().transfer_dth(boost::proto::child_c<2>(a0), n, queues[j], j , leftover );
    boost::proto::value(boost::proto::child_c<3>(a0)).specifics().transfer_dth(boost::proto::child_c<3>(a0), n, queues[j], j , leftover );
  }


  delete [] queues;


} // kernel
} // namespace nt2
