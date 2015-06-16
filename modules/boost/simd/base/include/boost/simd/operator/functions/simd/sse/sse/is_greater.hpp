//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_OPERATOR_FUNCTIONS_SIMD_SSE_SSE_IS_GREATER_HPP_INCLUDED
#define BOOST_SIMD_OPERATOR_FUNCTIONS_SIMD_SSE_SSE_IS_GREATER_HPP_INCLUDED
#ifdef BOOST_SIMD_HAS_SSE_SUPPORT

#include <boost/simd/operator/functions/is_greater.hpp>
#include <boost/simd/swar/functions/details/shuffle.hpp>
#include <boost/simd/include/functions/simd/bitwise_cast.hpp>
#include <boost/simd/include/functions/simd/logical_and.hpp>
#include <boost/simd/include/functions/simd/logical_or.hpp>
#include <boost/simd/include/functions/simd/is_equal.hpp>
#include <boost/simd/include/functions/simd/minus.hpp>
#include <boost/simd/include/constants/signmask.hpp>
#include <boost/simd/sdk/meta/as_logical.hpp>
#include <boost/dispatch/meta/as_integer.hpp>
#include <boost/dispatch/meta/downgrade.hpp>
#include <boost/dispatch/attributes.hpp>

namespace boost { namespace simd { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT          ( is_greater_
                                    , boost::simd::tag::sse_
                                    , (A0)
                                    , ((simd_<single_<A0>,boost::simd::tag::sse_>))
                                      ((simd_<single_<A0>,boost::simd::tag::sse_>))
                                    )
  {
    typedef typename meta::as_logical<A0>::type result_type;

    BOOST_FORCEINLINE BOOST_SIMD_FUNCTOR_CALL_REPEAT(2)
    {
      return _mm_cmpgt_ps(a0,a1);
    }
  };

} } }

#endif
#endif
