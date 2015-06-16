//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_OPERATOR_FUNCTIONS_SIMD_SSE_SSE_MULTIPLIES_HPP_INCLUDED
#define BOOST_SIMD_OPERATOR_FUNCTIONS_SIMD_SSE_SSE_MULTIPLIES_HPP_INCLUDED
#ifdef BOOST_SIMD_HAS_SSE_SUPPORT

#include <boost/simd/operator/functions/multiplies.hpp>
#include <boost/simd/include/constants/int_splat.hpp>
#include <boost/dispatch/meta/upgrade.hpp>
#include <boost/dispatch/attributes.hpp>

namespace boost { namespace simd { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT          ( multiplies_
                                    , boost::simd::tag::sse_
                                    , (A0)
                                    , ((simd_<single_<A0>,boost::simd::tag::sse_>))
                                      ((simd_<single_<A0>,boost::simd::tag::sse_>))
                                    )
  {
    typedef A0 result_type;

    BOOST_FORCEINLINE BOOST_SIMD_FUNCTOR_CALL_REPEAT(2)
    {
      return _mm_mul_ps(a0,a1);
    }
  };

} } }

#endif
#endif
