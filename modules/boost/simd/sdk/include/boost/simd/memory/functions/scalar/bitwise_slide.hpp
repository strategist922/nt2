//==============================================================================
//         Copyright 2015 NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_MEMORY_FUNCTIONS_BITWISE_SCALAR_SLIDE_HPP_INCLUDED
#define BOOST_SIMD_MEMORY_FUNCTIONS_BITWISE_SCALAR_SLIDE_HPP_INCLUDED

#include <boost/simd/memory/functions/bitwise_slide.hpp>
#include <boost/simd/include/functions/shift_left.hpp>
#include <boost/simd/include/functions/shift_right.hpp>
#include <boost/dispatch/functor/preprocessor/call.hpp>
#include <boost/dispatch/meta/mpl.hpp>
#include <boost/dispatch/attributes.hpp>

namespace boost { namespace simd { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT          ( bitwise_slide_
                                    , boost::simd::tag::cpu_
                                    , (A0)(N)
                                    , (scalar_< unsigned_<A0> >)
                                      (mpl_integral_< scalar_< integer_<N> > >)
                                    )
  {
    typedef A0 result_type;

    BOOST_FORCEINLINE result_type operator()(A0 a0, N const&) const
    {
      return N::value>=0 ? shift_left(a0,N::value)
                         : shift_right(a0,-N::value);
    }
  };

  BOOST_DISPATCH_IMPLEMENT          ( bitwise_slide_
                                    , boost::simd::tag::cpu_
                                    , (A0)(N)
                                    , (scalar_< unsigned_<A0> >)
                                      (scalar_< unsigned_<A0> >)
                                      (mpl_integral_< scalar_< integer_<N> > >)
                                    )
  {
    typedef A0 result_type;

    BOOST_FORCEINLINE result_type operator()(A0 a0, A0 a1, N const&) const
    {
      return N::value >= 0 ? a0<<N::value | a1 >> (sizeof(A0)*CHAR_BIT-N::value)
              : bitwise_slide<N::value>(a0);
    }
  };
} } }

#endif
