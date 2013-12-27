//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_EXPONENTIAL_FUNCTIONS_GENERIC_FAST_LOG10_HPP_INCLUDED
#define NT2_EXPONENTIAL_FUNCTIONS_GENERIC_FAST_LOG10_HPP_INCLUDED

#include <nt2/exponential/functions/fast_log10.hpp>
#include <nt2/exponential/functions/scalar/impl/fast_logs.hpp>
#include <nt2/exponential/functions/simd/common/impl/fast_logs.hpp>
#include <nt2/exponential/functions/scalar/impl/logs.hpp>
#include <nt2/exponential/functions/simd/common/impl/logs.hpp>
#include <nt2/include/functions/simd/frexp.hpp>
#include <boost/simd/sdk/simd/meta/is_native.hpp>

namespace nt2 { namespace ext
{

  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::fast_log10_, tag::cpu_
                            , (A0)
                            , (generic_< single_<A0> >)
                            )
  {
    typedef A0 result_type;
    typedef typename boost::simd::meta::is_native<A0>::type is_native_t;
    BOOST_FORCEINLINE NT2_FUNCTOR_CALL(1)
    {
      return details::fast_logarithm<A0,is_native_t>::log10(a0);
    }
  };

  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::fast_log10_, tag::cpu_
                            , (A0)
                            , (generic_< double_<A0> >)
                            )
  {
    typedef A0 result_type;
    typedef typename boost::simd::meta::is_native<A0>::type is_native_t;
    BOOST_FORCEINLINE NT2_FUNCTOR_CALL(1)
    {
      return details::logarithm<A0,is_native_t>::log10(a0);
    }
  };

} }


#endif
