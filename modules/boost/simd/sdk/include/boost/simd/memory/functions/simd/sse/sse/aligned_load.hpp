//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_MEMORY_FUNCTIONS_SIMD_SSE_SSE_ALIGNED_LOAD_HPP_INCLUDED
#define BOOST_SIMD_MEMORY_FUNCTIONS_SIMD_SSE_SSE_ALIGNED_LOAD_HPP_INCLUDED
#ifdef BOOST_SIMD_HAS_SSE_SUPPORT

#include <boost/simd/memory/functions/aligned_load.hpp>
#include <boost/simd/memory/functions/details/check_ptr.hpp>
#include <boost/dispatch/functor/preprocessor/call.hpp>
#include <boost/simd/include/functions/simd/bitwise_cast.hpp>
#include <boost/simd/include/functions/simd/slide.hpp>
#include <boost/simd/meta/is_pointing_to.hpp>
#include <boost/simd/sdk/simd/category.hpp>
#include <boost/dispatch/meta/mpl.hpp>
#include <boost/dispatch/attributes.hpp>

#ifdef BOOST_MSVC
#pragma warning(push)
#pragma warning(disable: 4308) // negative integral conversion
#endif

namespace boost { namespace simd { namespace ext
{
  /// INTERNAL ONLY - Load register of SIMD float
  BOOST_DISPATCH_IMPLEMENT          ( aligned_load_
                                    , boost::simd::tag::sse_
                                    , (A0)(A2)
                                    , (iterator_< scalar_< single_<A0> > >)
                                      ((target_ < simd_ < single_<A2>
                                                        , boost::simd::tag::sse_
                                                        >
                                                >
                                      ))
                                    )
  {
    typedef typename A2::type result_type;

    BOOST_FORCEINLINE
    result_type operator()(A0 a0, const A2&) const
    {
      BOOST_SIMD_DETAILS_CHECK_PTR(a0, sizeof(result_type));
      return _mm_load_ps(a0);
    }
  };

} } }

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif
#endif
