//==============================================================================
//         Copyright 2003 - 2013   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <boost/simd/ieee/include/functions/bitofsign.hpp>

#include <boost/dispatch/functor/meta/call.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/tests/type_expr.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <boost/simd/sdk/config.hpp>

#include <boost/simd/include/constants/mone.hpp>
#include <boost/simd/include/constants/mzero.hpp>
#include <boost/simd/include/constants/one.hpp>
#include <boost/simd/include/constants/zero.hpp>
#include <boost/simd/include/constants/inf.hpp>
#include <boost/simd/include/constants/minf.hpp>

NT2_TEST_CASE_TPL ( bitofsign_real,  BOOST_SIMD_REAL_TYPES)
{
  using boost::simd::bitofsign;
  using boost::simd::tag::bitofsign_;
  typedef typename boost::dispatch::meta::call<bitofsign_(T)>::type r_t;
  typedef T wished_r_t;

  // return type conformity test
  NT2_TEST_TYPE_IS(r_t, wished_r_t);

  // specific values tests
#ifndef BOOST_SIMD_NO_INVALIDS
  NT2_TEST_EQUAL(bitofsign(boost::simd::Inf<T>()), boost::simd::Zero<r_t>());
  NT2_TEST_EQUAL(bitofsign(boost::simd::Minf<T>()), boost::simd::Mzero<r_t>());
#endif
  NT2_TEST_EQUAL(bitofsign(boost::simd::Mzero<T>()), boost::simd::Mzero<r_t>());
  NT2_TEST_EQUAL(bitofsign(boost::simd::One<T>()), boost::simd::Zero<r_t>());
  NT2_TEST_EQUAL(bitofsign(boost::simd::Zero<T>()), boost::simd::Zero<r_t>());
}

NT2_TEST_CASE_TPL ( bitofsign_unsigned_int,  BOOST_SIMD_UNSIGNED_TYPES)
{
  using boost::simd::bitofsign;
  using boost::simd::tag::bitofsign_;
  typedef typename boost::dispatch::meta::call<bitofsign_(T)>::type r_t;
  typedef T wished_r_t;

  // return type conformity test
  NT2_TEST_TYPE_IS(r_t, wished_r_t);

  // specific values tests
  NT2_TEST_EQUAL(bitofsign(boost::simd::One<T>()), boost::simd::Zero<r_t>());
}

NT2_TEST_CASE_TPL ( bitofsign_signed_int,  BOOST_SIMD_INTEGRAL_SIGNED_TYPES)
{
  using boost::simd::bitofsign;
  using boost::simd::tag::bitofsign_;
  typedef typename boost::dispatch::meta::call<bitofsign_(T)>::type r_t;
  typedef T wished_r_t;

  // return type conformity test
  NT2_TEST_TYPE_IS(r_t, wished_r_t);

  // specific values tests
  NT2_TEST_EQUAL(bitofsign(boost::simd::Mone<T>()), T(1ull << (sizeof(T)*8-1)));
  NT2_TEST_EQUAL(bitofsign(boost::simd::One<T>()), boost::simd::Zero<r_t>());
  NT2_TEST_EQUAL(bitofsign(boost::simd::Zero<T>()), boost::simd::Zero<r_t>());
}
