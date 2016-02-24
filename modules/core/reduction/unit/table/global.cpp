//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/include/functions/of_size.hpp>
#include <nt2/include/functions/global.hpp>
#include <nt2/include/constants/true.hpp>
#include <nt2/include/functions/all.hpp>
#include <nt2/include/functions/sum.hpp>
#include <nt2/include/functions/asump.hpp>
#include <nt2/include/functions/reshape.hpp>
#include <nt2/table.hpp>

#include <nt2/sdk/unit/module.hpp>
#include <nt2/sdk/unit/tests/basic.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/type_expr.hpp>
#include <nt2/sdk/unit/tests/exceptions.hpp>

NT2_TEST_CASE_TPL( global, NT2_REAL_TYPES )
{
  nt2::table<T> a = nt2::reshape(nt2::_(T(1), T(9)), 3, 3);

  nt2::functor<nt2::tag::sum_>    sum_;
  nt2::functor<nt2::tag::asump_>  asump_;
  nt2::functor<nt2::tag::all_>    all_;

  NT2_TEST_EQUAL( nt2::global(asump_, a    , T(2)), T(285));
  NT2_TEST_EQUAL( nt2::global(asump_, T(42), T(2)), T(1764));

  NT2_TEST_EQUAL( nt2::global(sum_, a), T(45));
  NT2_TEST_EQUAL( nt2::global(sum_, T(42)), T(42));

  NT2_TEST_EQUAL( nt2::global(all_, a), true);
  a(3, 3) = T(0);
  NT2_TEST_EQUAL( nt2::global(all_, a), false);

  NT2_TEST_EQUAL( nt2::global(all_, T(42)), true);
  NT2_TEST_EQUAL( nt2::global(all_, T(0)), false);
}
