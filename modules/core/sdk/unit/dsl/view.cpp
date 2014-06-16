//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/table.hpp>
#include <nt2/include/functions/times.hpp>
#include <boost/array.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/type_expr.hpp>

NT2_TEST_CASE( view_over_pointer )
{
  using nt2::table;
  using nt2::of_size;
  using nt2::view;

  boost::array<double,12> data =  {{ 1., 2., 3., 4. , 5. , 6.
                                   , 7., 8., 9., 10., 11., 12.
                                  }};

  boost::array<double,12> ref =  {{ 10., 20., 30., 40. , 50. , 60.
                                   ,70., 80., 90., 100., 110., 120.
                                  }};

  view< table<double> > v(&data[0], of_size(3,4));
  NT2_TEST_EQUAL(data, v);

  v.reset(&data[0], of_size(6,2));
  NT2_TEST_EQUAL(data, v);

  v = 10. * v;
  NT2_TEST_EQUAL(v, ref);
  NT2_TEST_EQUAL(data, ref);
}
