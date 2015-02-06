#define NT2_UNIT_MODULE "nt2 future gpu"

#include <nt2/table.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/zeros.hpp>
#include <nt2/sdk/cuda/future.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/type_expr.hpp>
#include <nt2/sdk/unit/tests/exceptions.hpp>
#include "Obj_cuda.hpp"
#include <nt2/sdk/cuda/cuda.hpp>
#include <nt2/sdk/memory/buffer.hpp>
#include <iostream>
#include <vector>

using Site = typename boost::dispatch::default_site<void>::type;
using Arch = typename nt2::tag::cuda_<Site>;
using future_1 = typename nt2::make_future<Arch,int>::type;


NT2_TEST_CASE_TPL( future_gpu, (int) )
{
  nt2::table<T> x = nt2::ones(10,1, nt2::meta::as_<int>() );
  nt2::table<T> res = nt2::zeros(10,1, nt2::meta::as_<int>() );

  nt2::table<T,nt2::memory::cuda_buffer<T>> d_a = x;
  nt2::table<T,nt2::memory::cuda_buffer<T>> d_b = x;
  nt2::table<T,nt2::memory::cuda_buffer<T>> d_res = x;

  future_1 f1 = nt2::async<Arch>(Obj_cuda(),d_res.data(),d_a.data(),d_b.data());

  // test = f1.get();

  nt2::table<T> res1 = d_res;

  NT2_DISPLAY(res1);
 // NT2_TEST_EQUAL(10*(x+x), h_res);
}





