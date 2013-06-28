//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/reduction/include/functions/compare_greater_equal.hpp>
#include <boost/simd/sdk/simd/native.hpp>
#include <nt2/include/functions/all.hpp>

#include <boost/type_traits/is_same.hpp>
#include <nt2/sdk/functor/meta/call.hpp>
#include <nt2/sdk/meta/as_integer.hpp>
#include <nt2/sdk/meta/as_floating.hpp>
#include <nt2/sdk/meta/as_signed.hpp>
#include <nt2/sdk/meta/upgrade.hpp>
#include <nt2/sdk/meta/downgrade.hpp>
#include <nt2/sdk/meta/scalar_of.hpp>
#include <boost/dispatch/meta/as_floating.hpp>
#include <boost/type_traits/common_type.hpp>
#include <nt2/sdk/unit/tests.hpp>
#include <nt2/sdk/unit/module.hpp>
#include <nt2/constant/constant.hpp>
#include <nt2/sdk/meta/cardinal_of.hpp>
#include <nt2/include/functions/splat.hpp>

NT2_TEST_CASE_TPL ( compare_greater_equal_real__2_0,  NT2_SIMD_REAL_TYPES)
{
  using nt2::compare_greater_equal;
  using nt2::tag::compare_greater_equal_;
  using boost::simd::native;
  using nt2::meta::cardinal_of;
  typedef NT2_SIMD_DEFAULT_EXTENSION  ext_t;
  typedef typename nt2::meta::upgrade<T>::type   u_t;
  typedef native<T,ext_t>                        n_t;
  typedef n_t                                     vT;
  typedef typename nt2::meta::as_integer<T>::type iT;
  typedef native<iT,ext_t>                       ivT;
  typedef typename nt2::meta::call<compare_greater_equal_(vT,vT)>::type r_t;
  typedef typename nt2::meta::call<compare_greater_equal_(T,T)>::type sr_t;
  typedef typename nt2::meta::scalar_of<r_t>::type ssr_t;
  double ulpd;
  ulpd=0.0;


  // specific values tests
  NT2_TEST_EQUAL(compare_greater_equal(nt2::Inf<vT>(), nt2::Inf<vT>()), ssr_t(true));
  NT2_TEST_EQUAL(compare_greater_equal(nt2::Minf<vT>(), nt2::Minf<vT>()), ssr_t(true));
  NT2_TEST_EQUAL(compare_greater_equal(nt2::Nan<vT>(), nt2::Nan<vT>()), ssr_t(true));
  NT2_TEST_EQUAL(compare_greater_equal(nt2::One<vT>(),nt2::Zero<vT>()), ssr_t(true));
  NT2_TEST_EQUAL(compare_greater_equal(nt2::Zero<vT>(), nt2::Zero<vT>()), ssr_t(true));
} // end of test for floating_

NT2_TEST_CASE_TPL ( compare_greater_equal_signed_int__2_0,  NT2_SIMD_INTEGRAL_SIGNED_TYPES)
{
  using nt2::compare_greater_equal;
  using nt2::tag::compare_greater_equal_;
  using boost::simd::native;
  using nt2::meta::cardinal_of;
  typedef NT2_SIMD_DEFAULT_EXTENSION  ext_t;
  typedef typename nt2::meta::upgrade<T>::type   u_t;
  typedef native<T,ext_t>                        n_t;
  typedef n_t                                     vT;
  typedef typename nt2::meta::as_integer<T>::type iT;
  typedef native<iT,ext_t>                       ivT;
  typedef typename nt2::meta::call<compare_greater_equal_(vT,vT)>::type r_t;
  typedef typename nt2::meta::call<compare_greater_equal_(T,T)>::type sr_t;
  typedef typename nt2::meta::scalar_of<r_t>::type ssr_t;
  double ulpd;
  ulpd=0.0;


  // specific values tests
  NT2_TEST_EQUAL(compare_greater_equal(nt2::Mone<vT>(),nt2::Zero<vT>()), ssr_t(false));
  NT2_TEST_EQUAL(compare_greater_equal(nt2::One<vT>(), nt2::One<vT>()), ssr_t(true));
  NT2_TEST_EQUAL(compare_greater_equal(nt2::One<vT>(),nt2::Zero<vT>()), ssr_t(true));
  NT2_TEST_EQUAL(compare_greater_equal(nt2::Zero<vT>(), nt2::Zero<vT>()), ssr_t(true));
} // end of test for signed_int_

NT2_TEST_CASE_TPL ( compare_greater_equal_unsigned_int__2_0,  NT2_SIMD_UNSIGNED_TYPES)
{
  using nt2::compare_greater_equal;
  using nt2::tag::compare_greater_equal_;
  using boost::simd::native;
  using nt2::meta::cardinal_of;
  typedef NT2_SIMD_DEFAULT_EXTENSION  ext_t;
  typedef typename nt2::meta::upgrade<T>::type   u_t;
  typedef native<T,ext_t>                        n_t;
  typedef n_t                                     vT;
  typedef typename nt2::meta::as_integer<T>::type iT;
  typedef native<iT,ext_t>                       ivT;
  typedef typename nt2::meta::call<compare_greater_equal_(vT,vT)>::type r_t;
  typedef typename nt2::meta::call<compare_greater_equal_(T,T)>::type sr_t;
  typedef typename nt2::meta::scalar_of<r_t>::type ssr_t;
  double ulpd;
  ulpd=0.0;


  // specific values tests
  NT2_TEST_EQUAL(compare_greater_equal(nt2::One<vT>(), nt2::One<vT>()), ssr_t(true));
  NT2_TEST_EQUAL(compare_greater_equal(nt2::One<vT>(),nt2::Zero<vT>()), ssr_t(true));
  NT2_TEST_EQUAL(compare_greater_equal(nt2::Zero<vT>(), nt2::Zero<vT>()), ssr_t(true));
} // end of test for unsigned_int_
