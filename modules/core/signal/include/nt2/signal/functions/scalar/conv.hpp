//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SIGNAL_FUNCTIONS_SCALAR_CONV_HPP_INCLUDED
#define NT2_SIGNAL_FUNCTIONS_SCALAR_CONV_HPP_INCLUDED

#include <nt2/signal/functions/conv.hpp>
#include <nt2/include/functions/multiplies.hpp>

namespace nt2 { namespace ext
{
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::conv_, boost::simd::tag::simd_
                            , (A0)(Shp)
                            , (scalar_< unspecified_<A0> >)
                              (scalar_< unspecified_<A0> >)
                              (unspecified_< Shp >)
                            )
  {
    typedef A0 result_type;

    BOOST_FORCEINLINE
    result_type operator()(A0 const& a0, A0 const& a1, Shp const&) const
    {
      return nt2::multiplies(a0,a1);
    }
  };
} }

#endif
