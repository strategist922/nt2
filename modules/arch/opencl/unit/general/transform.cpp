//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <iostream>

#include <nt2/table.hpp>

//#include <nt2/include/functions/zeros.hpp>
//#include <nt2/include/functions/ones.hpp>
//#include <nt2/include/functions/size.hpp>

#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/module.hpp>

#include <nt2/sdk/meta/type_id.hpp>

// #include <nt2/include/functions/tie.hpp>
#include <nt2/core/functions/transform.hpp>
#include <nt2/include/functions/is_equal.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>

#include <nt2/include/functions/log.hpp>
#include <nt2/include/functions/divides.hpp>
#include <nt2/include/functions/minus.hpp>
#include <nt2/include/functions/plus.hpp>

#include <time.h>
#include "generated.cpp"

namespace compute = boost::compute;


//NT2_TEST_CASE_TPL( opencl_buffer_swap, (double) )
//{
//  nt2::table<T> A_init = nt2::ones(10,1,nt2::meta::as_<T>());
//  nt2::table<T> B_init = nt2::zeros(10,1,nt2::meta::as_<T>());
//
//  nt2::table<T,nt2::device_> cl_A, cl_B;
//  cl_A = A_init;
//  cl_B = B_init;
//
//  nt2::table<T> A_final, B_final;
//  A_final = cl_B;
//  B_final = cl_A;
//
//  NT2_TEST_EQUAL(A_final,B_init);
//  NT2_TEST_EQUAL(B_final,A_init);
//}

NT2_TEST_CASE_TPL( direct_transform, (float) )
{
 using nt2::of_size;
 std::size_t x = 8000000;
 std::size_t y = 1;

 nt2::table<T> out;
 nt2::table<T, nt2::device_> Sa(of_size(x,y)), Xa(of_size(x,y));

//  for(std::size_t i=1;i<=nt2::numel(Sa);++i)
//  {
//   Sa(i) = Xa(i) = Ta(i) = T(i);
//   out(i) = T(0);
//   }
   
//out = Xa;
//Xa = out;

// out = 
   nt2::log(Sa/Xa) + Xa;
//nt2::log(Sa);

// NT2_TEST_EQUAL(T(0), T(0));
 NT2_TEST_EQUAL(1, 1);

}

