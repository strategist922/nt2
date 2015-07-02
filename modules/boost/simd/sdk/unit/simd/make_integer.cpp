//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#define NT2_UNIT_MODULE "boost::simd::meta::make_integer SIMD"

#include <boost/simd/sdk/simd/pack.hpp>
#include <boost/simd/sdk/simd/meta/vector_of.hpp>
#include <boost/simd/sdk/simd/meta/native_cardinal.hpp>
#include <boost/dispatch/meta/make_integer.hpp>
#include <boost/type_traits/is_same.hpp>

#include <nt2/sdk/unit/tests/type_expr.hpp>
#include <nt2/sdk/unit/module.hpp>

////////////////////////////////////////////////////////////////////////////////
// Test that make_integer on SIMD with unsigned target
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE_TPL(make_uinteger_simd_native, BOOST_SIMD_SIMD_TYPES)
{
  using boost::simd::meta::vector_of;
  using boost::simd::meta::native_cardinal;
  using boost::dispatch::meta::make_integer;

  typedef typename vector_of< typename make_integer<sizeof(T),unsigned>::type
                            , native_cardinal<T>::value
                            >::type                               dst_t;
  typedef typename boost::dispatch::meta::factory_of<dst_t>::type fact_t;

  NT2_TEST_TYPE_IS( (typename make_integer<sizeof(T),unsigned, fact_t>::type)
                  , dst_t
                  );
}

////////////////////////////////////////////////////////////////////////////////
// Test that make_integer on SIMD with signed target
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE_TPL(make_integer_simd_native, BOOST_SIMD_SIMD_TYPES)
{
  using boost::simd::meta::vector_of;
  using boost::simd::meta::native_cardinal;
  using boost::dispatch::meta::make_integer;

  typedef typename vector_of< typename make_integer<sizeof(T),signed>::type
                            , native_cardinal<T>::value
                            >::type                               dst_t;
  typedef typename boost::dispatch::meta::factory_of<dst_t>::type fact_t;

  NT2_TEST_TYPE_IS( (typename make_integer<sizeof(T),signed, fact_t>::type)
                  , dst_t
                  );
}

////////////////////////////////////////////////////////////////////////////////
// Test that make_integer on SIMD with unsigned target
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE_TPL(make_uinteger_simd_pack, BOOST_SIMD_SIMD_TYPES)
{
  using boost::simd::pack;
  using boost::dispatch::meta::make_integer;
  using boost::is_same;
  using boost::mpl::_;

  typedef pack<typename make_integer<sizeof(T),unsigned>::type>    dst_t;
  typedef typename boost::dispatch::meta::factory_of<dst_t>::type fact_t;

  NT2_TEST_TYPE_IS( (typename make_integer<sizeof(T),unsigned, fact_t>::type)
                  , dst_t
                  );
}

////////////////////////////////////////////////////////////////////////////////
// Test that make_integer on SIMD with signed target
////////////////////////////////////////////////////////////////////////////////
NT2_TEST_CASE_TPL(make_integer_simd_pack, BOOST_SIMD_SIMD_TYPES)
{
  using boost::simd::pack;
  using boost::dispatch::meta::make_integer;
  using boost::is_same;
  using boost::mpl::_;

  typedef pack<typename make_integer<sizeof(T),signed>::type> dst_t;
  typedef typename boost::dispatch::meta::factory_of<dst_t>::type fact_t;

  NT2_TEST_TYPE_IS( (typename make_integer<sizeof(T),signed, fact_t>::type)
                  , dst_t
                  );
}
