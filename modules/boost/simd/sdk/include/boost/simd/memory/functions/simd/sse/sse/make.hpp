//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_MEMORY_FUNCTIONS_SIMD_SSE_SSE_MAKE_HPP_INCLUDED
#define BOOST_SIMD_MEMORY_FUNCTIONS_SIMD_SSE_SSE_MAKE_HPP_INCLUDED
#ifdef BOOST_SIMD_HAS_SSE_SUPPORT

#include <boost/simd/memory/functions/make.hpp>
#include <boost/simd/preprocessor/make_helper.hpp>
#include <boost/dispatch/attributes.hpp>

namespace boost { namespace simd { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT          ( make_
                                    , boost::simd::tag::sse_
                                    , (A0)
                                    , ((target_ < simd_ < single_<A0>
                                                        , boost::simd::tag::sse_
                                                        >
                                                >
                                      ))
                                    )
  {
    BOOST_SIMD_MAKE_BODY(4) { return _mm_setr_ps(a0, a1, a2, a3); }
  };

} } }

#endif
#endif
