//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/sdk/cuda/cuda.hpp>
#include <nt2/sdk/memory/buffer.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/size.hpp>
#include <nt2/table.hpp>
#include <cublas.h>
#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>



// NT2_TEST_CASE_TPL( cuda_buffer_d, (double) )
// {
//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > entry(nt2::of_size(5,1));


//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > cuda_dst(nt2::of_size(5,1));
//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > result(nt2::of_size(5,1));

//   for(auto i : {0,1,2,3,4})  entry[i] = 1. ;
//   for(auto i : {0,1,2,3,4})  result[i] = 5. ;

//   T alpha = 5.;
//   int incr =1;

//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > test(nt2::of_size(5,2,2));
//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > test1(test);
//   nt2::memory::container<nt2::tag::table_,T,nt2::memory::cuda_buffer<T>> cudabuffer(entry) ;

//   cublasDscal( cudabuffer.size(), alpha ,cudabuffer.data(), incr);

//   CUDA_ERROR(cudaMemcpy( cuda_dst.data()
//                                 , cudabuffer.data()
//                                 , cudabuffer.size() * sizeof(double)
//                                 , cudaMemcpyDeviceToHost
//                                 ));

//   for(auto i : cuda_dst) std::cout << i << std::endl;
// }

// NT2_TEST_CASE_TPL( cuda_buffer_copy_d, (double) )
// {
//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > entry(nt2::of_size(5,1));


//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > cuda_dst(nt2::of_size(5,1));
//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > result(nt2::of_size(5,1));

//   for(auto i : {0,1,2,3,4})  entry[i] = 1. ;
//   for(auto i : {0,1,2,3,4})  result[i] = 5. ;

//   T alpha = 5.;
//   int incr =1;

//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > test(nt2::of_size(5,2,2));
//   nt2::memory::container<nt2::tag::table_,T,nt2::_2D > test1(test);
//   nt2::memory::container<nt2::tag::table_,T,nt2::memory::cuda_buffer<T>> cudabuffer(entry) ;

//   cublasDscal( cudabuffer.size(), alpha ,cudabuffer.data(), incr);

//   nt2::memory::copy(boost::proto::value(cudabuffer),cuda_dst);

//   for(auto i : cuda_dst) std::cout << i << std::endl;


//   // NT2_TEST_EQUAL(result, cuda_dst );
//   // NT2_TEST_EQUAL(cudabuffer.size(), entry.size() );
// }



NT2_TEST_CASE_TPL( cuda_buffer_table, (double) )
{

  nt2::table<T> x = nt2::ones(nt2::of_size(5,1), nt2::meta::as_<T>() );
  nt2::table<T,nt2::memory::cuda_buffer<T>> y = x;

  T alpha = 5.;
  int incr =1;

  cublasDscal( y.size(), alpha ,y.data(), incr);

  CUDA_ERROR(cudaMemcpy( x.data()
                                , y.data()
                                , y.size() * sizeof(double)
                                , cudaMemcpyDeviceToHost
                                ));


  NT2_DISPLAY(x);
}


NT2_TEST_CASE_TPL( cuda_buffer_table1, (double) )
{

  nt2::table<T> x = nt2::ones(nt2::of_size(5,1), nt2::meta::as_<T>() );
  nt2::table<T,nt2::memory::cuda_buffer<T>> y = x;

  T alpha = 5.;
  int incr =1;

  cublasDscal( y.size(), alpha ,y.data(), incr);

  nt2::table<T> res = y;

  NT2_DISPLAY(res);
}


NT2_TEST_CASE_TPL( cuda_buffer_table2, (double) )
{
  nt2::table<T> x = nt2::ones(nt2::of_size(5,1), nt2::meta::as_<T>() );
  nt2::table<T,nt2::memory::cuda_buffer<T>> y = x;

  T alpha = 5.;
  int incr =1;

  cublasDscal( y.size(), alpha ,y.data(), incr);

  nt2::table<T> res;
  res = y;

  NT2_DISPLAY(res);
}
