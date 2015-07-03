//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_BITWISE_FUNCTIONS_SIMD_SSE_SSE_ROR_HPP_INCLUDED
#define BOOST_SIMD_BITWISE_FUNCTIONS_SIMD_SSE_SSE_ROR_HPP_INCLUDED

#include <boost/simd/bitwise/functions/ror.hpp>
#include <boost/simd/include/functions/simd/splat.hpp>
#include <boost/simd/sdk/meta/cardinal_of.hpp>
#include <boost/simd/sdk/meta/scalar_of.hpp>
#include <boost/dispatch/meta/as_integer.hpp>
#include <boost/mpl/equal_to.hpp>
#include <boost/dispatch/attributes.hpp>

namespace boost { namespace simd { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT             ( ror_, tag::sse_
                                       , (A0)(A1)
                                       , ((simd_< floating_<A0>, tag::sse_ >))
                                         (scalar_< integer_<A1> >)
                                       )
  {
    typedef A0 result_type;

    BOOST_FORCEINLINE result_type operator()(A0 const& a0, A1 const& a1) const
    {
      typedef typename meta::vector_of<A1, A0::static_size>::type iA0;
      return ror(a0, boost::simd::splat<iA0>(a1));
    }
  };
} } }

#endif
