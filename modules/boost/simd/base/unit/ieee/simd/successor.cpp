//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <boost/simd/ieee/include/functions/successor.hpp>
#include <boost/simd/sdk/simd/native.hpp>
#include <nt2/sdk/unit/tests/relation.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <boost/simd/sdk/config.hpp>
#include <boost/simd/sdk/simd/io.hpp>

#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/minus.hpp>
#include <boost/simd/include/functions/unary_minus.hpp>
#include <boost/simd/include/constants/bitincrement.hpp>
#include <boost/simd/include/constants/eps.hpp>
#include <boost/simd/include/constants/halfeps.hpp>
#include <boost/simd/include/constants/mone.hpp>
#include <boost/simd/include/constants/one.hpp>
#include <boost/simd/include/constants/two.hpp>
#include <boost/simd/include/constants/four.hpp>
#include <boost/simd/include/constants/valmax.hpp>
#include <boost/simd/include/constants/valmin.hpp>
#include <boost/simd/include/constants/zero.hpp>
#include <boost/simd/include/constants/inf.hpp>
#include <boost/simd/include/constants/minf.hpp>
#include <boost/simd/include/constants/nan.hpp>

NT2_TEST_CASE_TPL ( successor_real_1,  BOOST_SIMD_SIMD_REAL_TYPES)
{
  using boost::simd::successor;
  using boost::simd::tag::successor_;
  using boost::simd::native;
  typedef BOOST_SIMD_DEFAULT_EXTENSION  ext_t;
  typedef native<T,ext_t>                  vT;
  typedef typename boost::dispatch::meta::call<successor_(vT)>::type r_t;

  // specific values tests
#ifndef BOOST_SIMD_NO_INVALIDS
  NT2_TEST_EQUAL(successor(boost::simd::Inf<vT>()), boost::simd::Nan<r_t>());
  NT2_TEST_EQUAL(successor(boost::simd::Minf<vT>()), boost::simd::Valmin<r_t>());
  NT2_TEST_EQUAL(successor(boost::simd::Nan<vT>()), boost::simd::Nan<r_t>());
#endif
  NT2_TEST_EQUAL(successor(boost::simd::Mone<vT>()), boost::simd::Mone<r_t>()+boost::simd::Halfeps<r_t>());
  NT2_TEST_EQUAL(successor(boost::simd::One<vT>()), boost::simd::One<r_t>()+boost::simd::Eps<r_t>());
  NT2_TEST_EQUAL(successor(boost::simd::Valmax<vT>()), boost::simd::Inf<r_t>());
#if !defined(BOOST_SIMD_NO_DENORMALS)
  NT2_TEST_EQUAL(successor(boost::simd::Zero<vT>()), boost::simd::Bitincrement<r_t>());
#endif
} // end of test for floating_

NT2_TEST_CASE_TPL ( successor_real_2,  BOOST_SIMD_SIMD_REAL_TYPES)
{
  using boost::simd::meta::vector_of;
  using boost::simd::successor;
  using boost::simd::tag::successor_;
  using boost::simd::native;
  typedef BOOST_SIMD_DEFAULT_EXTENSION  ext_t;
  typedef native<T,ext_t>                  vT;
  typedef typename boost::dispatch::meta::as_integer<T>::type iT;
  typedef typename vector_of<iT, vT::static_size>::type       ivT;
  typedef typename boost::dispatch::meta::call<successor_(vT, ivT)>::type r_t;

  // specific values tests
#ifndef BOOST_SIMD_NO_INVALIDS
  NT2_TEST_EQUAL(successor(boost::simd::Inf<vT>(), boost::simd::Two<ivT>()), boost::simd::Nan<r_t>());
  NT2_TEST_EQUAL(successor(boost::simd::Nan<vT>(), boost::simd::Two<ivT>()), boost::simd::Nan<r_t>());
#endif
  NT2_TEST_EQUAL(successor(boost::simd::Mone<vT>(), boost::simd::Two<ivT>()), boost::simd::Mone<r_t>()+boost::simd::Eps<r_t>());
  NT2_TEST_EQUAL(successor(boost::simd::One<vT>(), boost::simd::Two<ivT>()), boost::simd::One<r_t>()+boost::simd::Eps<r_t>()+boost::simd::Eps<r_t>());
  NT2_TEST_EQUAL(successor(boost::simd::Valmax<vT>(), boost::simd::Two<ivT>()), boost::simd::Nan<r_t>());
#if !defined(BOOST_SIMD_NO_DENORMALS)
  NT2_TEST_EQUAL(successor(boost::simd::Zero<vT>(), boost::simd::Two<ivT>()), boost::simd::Bitincrement<r_t>()+boost::simd::Bitincrement<r_t>());
#endif
} // end of test for floating_
