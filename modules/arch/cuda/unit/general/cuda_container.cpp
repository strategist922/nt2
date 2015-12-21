//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/sdk/cuda/cuda.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/size.hpp>
#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/table.hpp>
#include <cublas.h>


NT2_TEST_CASE_TPL( cuda_buffer_table1, (double) )
{

  nt2::table<T> x = nt2::ones(nt2::of_size(5,1), nt2::meta::as_<T>() );
  nt2::table<T> result = T(5)*nt2::ones(nt2::of_size(5,1), nt2::meta::as_<T>() );
  nt2::table<T,nt2::device_> y = x;

  T alpha = 5.;
  int incr =1;

  cublasDscal( y.size(), alpha ,y.data(), incr);

  nt2::table<T> res = y;

  NT2_TEST_EQUAL(res,result);
}

NT2_TEST_CASE_TPL( cuda_buffer_init, (double) )
{
  nt2::table<T> x = nt2::ones(nt2::of_size(5,1), nt2::meta::as_<T>() );
  nt2::table<T> result = T(5)*nt2::ones(nt2::of_size(5,1), nt2::meta::as_<T>() );
  nt2::table<T,nt2::device_ > y(nt2::of_size(5,1));
  y = x;

  T alpha = 5.;
  int incr =1;

  cublasDscal( y.size(), alpha ,y.data(), incr);

  nt2::table<T> res;
  res = y;

NT2_TEST_EQUAL(res,result);
}


NT2_TEST_CASE_TPL( cuda_buffer_device, (double) )
{
  nt2::table<T> x = nt2::ones(nt2::of_size(5,1), nt2::meta::as_<T>() );
  nt2::table<T> result = T(5)*nt2::ones(nt2::of_size(5,1), nt2::meta::as_<T>() );
  nt2::table<T,nt2::device_> y1 = x;
  nt2::table<T,nt2::device_> y = y1;

  T alpha = 5.;
  int incr =1;

  cublasDscal( y.size(), alpha ,y.data(), incr);

  nt2::table<T> res;
  res = y;

NT2_TEST_EQUAL(res,result);
}


