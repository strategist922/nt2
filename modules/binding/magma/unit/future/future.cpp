#define NT2_UNIT_MODULE "nt2 future gpu"

#include <nt2/table.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/zeros.hpp>
#include <nt2/sdk/magma/future.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/type_expr.hpp>
#include <nt2/sdk/unit/tests/exceptions.hpp>
#include "Obj_cuda.hpp"
#include <cublas.h>
#include <iostream>
#include <vector>

using Site = typename boost::dispatch::default_site<void>::type;
using Arch = typename nt2::tag::magma_<Site>;
using future_1 = typename nt2::make_future<Arch,int>::type;

NT2_TEST_CASE( future_get )
{
  nt2::table<int> x = nt2::ones(10,1, nt2::meta::as_<int>() );
  nt2::table<int> res = nt2::zeros(10,1, nt2::meta::as_<int>() );
  int size = sizeof(int)*10;
  int *d_res, *d_a, *d_b;

  cudaMalloc(  (void**) &d_a, size );
  cudaMalloc(  (void**) &d_b, size );
  cudaMalloc(  (void**) &d_res, size );

  cudaMemcpy(  d_a , x.raw() , size, cudaMemcpyHostToDevice);
  cudaMemcpy(  d_b , x.raw() , size, cudaMemcpyHostToDevice);
  cudaMemcpy(  d_res , res.raw() , size, cudaMemcpyHostToDevice);

  future_1 f1 = nt2::async<Arch>(Obj_cuda(),d_res,d_a,d_b,10);
  // res= f1.get();

  std::vector<int> h_res(10,0) ;
  cudaMemcpy(  h_res.data() , d_res , size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(  d_a , h_res.data()  , size, cudaMemcpyDeviceToHost);

  for(auto i : h_res)
    std::cout << i << std::endl;
  // NT2_TEST_EQUAL(x+x, res);
}
