//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <iostream>

#include <nt2/table.hpp>

#include <nt2/include/functions/zeros.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/size.hpp>
#include <nt2/core/functions/opencl/transform.hpp>

#include <nt2/sdk/unit/tests/ulp.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/module.hpp>

#include <nt2/include/functions/is_equal.hpp>

#include <nt2/include/functions/log.hpp>
#include <time.h>

#include <nt2/sdk/opencl/settings/specific_data.hpp>

namespace compute = boost::compute;


NT2_TEST_CASE_TPL( direct_transform, (float) )
{
 using nt2::of_size;
 std::size_t x = 800;
 std::size_t y = 1;

  typedef boost::dispatch::default_site<void>::type current_default_site;

  std::cout << nt2::type_id<current_default_site>() << "\n";

 nt2::table<T> out(of_size(x,y));
 nt2::table<T> in(of_size(x,y));
 nt2::table<T, nt2::device_> Sa(of_size(x,y)), Xa(of_size(x,y));

 for(std::size_t i=1;i<=nt2::numel(Sa);++i)
 {
//  Sa(i) = Xa(i) = T(i);
  out(i) = T(0);
  in(i) = T(3);
  }
  Xa = Sa = in;

//  out = Xa;
//  Xa = out;
//  Sa = out;

  out =
    nt2::log(Sa/Xa) + Xa;

// NT2_TEST_EQUAL(T(0), T(0));
 std::cout << /*Sa(1) << "\t" << Xa(1) << "\t" <<*/ out(1) << "\n";
 NT2_TEST_EQUAL(1, 1);
}
