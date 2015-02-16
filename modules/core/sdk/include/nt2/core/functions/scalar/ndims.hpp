//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SCALAR_NDIMS_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SCALAR_NDIMS_HPP_INCLUDED

#include <nt2/core/functions/ndims.hpp>
#include <boost/mpl/size_t.hpp>
#include <algorithm>

namespace nt2 { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT(ndims_, tag::cpu_, (A0), (scalar_<unspecified_<A0>>))
  {
    typedef boost::mpl::size_t<2> result_type;

    BOOST_FORCEINLINE result_type operator()(const A0&) const
    {
      return result_type();
    }
  };

  BOOST_DISPATCH_IMPLEMENT( ndims_, tag::cpu_
                          , (A0)(N)
                          , ((fusion_sequence_<A0,N>))
                          )
  {
    typedef std::size_t result_type;

    result_type operator()(A0& sz) const
    {
      // find the first non-1 from the end
      auto b = sz.rbegin(), e = sz.rend();
      auto c = std::find_if(b,e,[](std::size_t i) { return i != 1; });

      return std::distance(c,e);
    }
  };
} }

#endif
