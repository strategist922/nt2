//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#define NT2_UNIT_MODULE "nt2::transform function"

#include <nt2/sdk/opencl/settings/specific_data.hpp>
#include <nt2/table.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/zeros.hpp>
#include <nt2/include/functions/plus.hpp>
#include <nt2/include/functions/tie.hpp>
#include <nt2/core/functions/transform.hpp>
//#include <nt2/include/functions/tic.hpp>
//#include <nt2/include/functions/toc.hpp>
#include <nt2/include/functions/is_equal.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>

#include <nt2/include/functions/log.hpp>
#include <nt2/include/functions/exp.hpp>
#include <nt2/include/functions/fastnormcdf.hpp>
#include <nt2/include/functions/fma.hpp>
#include <nt2/include/functions/fnms.hpp>
#include <nt2/include/functions/sqrt.hpp>
#include <nt2/include/functions/sqr.hpp>
#include <nt2/include/functions/multiplies.hpp>
#include <nt2/include/functions/divides.hpp>
#include <nt2/include/functions/multiplies.hpp>
#include <nt2/include/functions/unary_minus.hpp>
#include <nt2/include/functions/minus.hpp>
#include <nt2/include/functions/plus.hpp>
#include <nt2/include/functions/tie.hpp>
#include <nt2/include/constants/half.hpp>

#include <time.h>

NT2_TEST_CASE_TPL( direct_transform, (float) )
{
  using nt2::of_size;
  std::size_t x = 8000000;
  std::size_t y = 1;

  const T ra = 0.02;
  const T va = 0.30;
  nt2::table<T> out( nt2::of_size(x,y) );
  nt2::table<T> Sa(of_size(x,y)), Xa(of_size(x,y)), Ta(of_size(x,y))
  , R(of_size(x,y)), da(of_size(x,y))
  , d1(of_size(x,y)), d2(of_size(x,y));

  // nt2::table<T, nt2::device_> in( nt2::of_size(x,y) );
  // nt2::table<T> in1( nt2::of_size(x,y) );
   //= nt2::ones( nt2::of_size(5,7), nt2::meta::as_<T>() );

  // int device;
  // cudaGetDevice(&device);

  // struct cudaDeviceProp props;
  // cudaGetDeviceProperties(&props, device);

  // std::cout << "props :" << std::endl;

  // std::cout << props.name << std::endl;


  double comp_time = 0;

   for(std::size_t i=1;i<=nt2::numel(Sa);++i)
   {
    Sa(i) = Xa(i) = Ta(i) = T(i);
    out(i) = T(0);
    //in(i) = T(1);
    //in1(i) = T(2);
   }

  // out = in1 + in;

//    nt2::tic();
   // clock_t start = clock();


   nt2::tie(da,d1,d2,R) = nt2::tie ( nt2::sqrt(Ta)
                                     , nt2::log(Sa/Xa) + (nt2::fma(nt2::sqr(va),T(0.5),ra)*Ta)/(va*da)
                                     , nt2::fnms(va,da,d1)
                                     , nt2::fnms(Xa*nt2::exp(-ra*Ta),nt2::fastnormcdf(d2),Sa*nt2::fastnormcdf(d1))
                                     );


//  comp_time = nt2::toc(false);

//  std::cout << " full time transform: " << comp_time << std::endl;
//    clock_t end = clock();
//    double time_elapsed_in_seconds = (end - start)/(double)CLOCKS_PER_SEC;
// std::cout << "clock time s : " << time_elapsed_in_seconds << std::endl;
//   auto rest = is_equal(da,out);
//   if(rest(1) == false )
//     {
//       std::cout << "result ok " << std::endl;
//     }
//   else
//     {
//       std::cout << "result wrong " << std::endl;
//     }


  // NT2_DISPLAY(da);
  // NT2_DISPLAY(d1);
  // NT2_DISPLAY(out);
  // NT2_DISPLAY(d2);
  // NT2_DISPLAY(R);
//    NT2_DISPLAY(out);
  // for(std::size_t i=1;i<=nt2::numel(out);++i)
  //   NT2_TEST_EQUAL(out(i), in(i)+in(i));


  NT2_TEST_EQUAL(T(0), T(0));

}
