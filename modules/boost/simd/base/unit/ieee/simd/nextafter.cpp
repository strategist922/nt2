//==============================================================================
//         Copyright 2003 - 2013   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <boost/simd/ieee/include/functions/nextafter.hpp>

#include <boost/dispatch/functor/meta/call.hpp>
#include <boost/simd/sdk/simd/native.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <boost/simd/sdk/config.hpp>
#include <boost/simd/sdk/simd/io.hpp>

#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/constants/mone.hpp>
#include <boost/simd/include/constants/one.hpp>
#include <boost/simd/include/constants/valmax.hpp>
#include <boost/simd/include/constants/zero.hpp>
#include <boost/simd/include/constants/inf.hpp>
#include <boost/simd/include/constants/minf.hpp>
#include <boost/simd/include/constants/nan.hpp>
#include <boost/simd/include/constants/bitincrement.hpp>
#include <boost/simd/include/constants/eps.hpp>
#include <boost/simd/include/constants/halfeps.hpp>
NT2_TEST_CASE_TPL ( nextafter_real,  BOOST_SIMD_SIMD_REAL_TYPES)
{
  using boost::simd::nextafter;
  using boost::simd::tag::nextafter_;
  using boost::simd::native;
  typedef BOOST_SIMD_DEFAULT_EXTENSION  ext_t;
  typedef native<T,ext_t>                  vT;
  typedef typename boost::dispatch::meta::call<nextafter_(vT,vT)>::type r_t;

  // specific values tests
#ifndef BOOST_SIMD_NO_INVALIDS
  NT2_TEST_EQUAL(nextafter(boost::simd::Inf<vT>(), boost::simd::Inf<vT>())   , boost::simd::Inf<r_t>()  );
  NT2_TEST_EQUAL(nextafter(boost::simd::Minf<vT>(), boost::simd::One<vT>())  , boost::simd::Valmin<r_t>() );
  NT2_TEST_EQUAL(nextafter(boost::simd::Nan<vT>(), boost::simd::One<vT>())   , boost::simd::Nan<r_t>() );
  NT2_TEST_EQUAL(nextafter(boost::simd::One<vT>(), boost::simd::Inf<vT>())   , boost::simd::One<r_t>()+boost::simd::Eps<r_t>());
  NT2_TEST_EQUAL(nextafter(boost::simd::Valmax<vT>(), boost::simd::Inf<vT>()), boost::simd::Inf<r_t>());
#endif
  NT2_TEST_EQUAL(nextafter(boost::simd::Mone<vT>(), boost::simd::One<vT>())  ,boost::simd::Mone<r_t>()+boost::simd::Halfeps<r_t>() );
  NT2_TEST_EQUAL(nextafter(boost::simd::Zero<vT>(), boost::simd::One<vT>())  ,boost::simd::Bitincrement<r_t>() );
}
