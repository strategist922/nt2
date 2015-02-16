//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SCALAR_FIRSTNONSINGLETON_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SCALAR_FIRSTNONSINGLETON_HPP_INCLUDED

#include <nt2/core/functions/firstnonsingleton.hpp>
#include <boost/mpl/size_t.hpp>

namespace nt2 { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT( firstnonsingleton_, tag::cpu_
                          , (A0)
                          , (scalar_< unspecified_<A0> >)
                          )
  {
    BOOST_DISPATCH_RETURNS_ARGS ( 1, (A0 const& a0), (A0 const& )
                                , boost::mpl::size_t<1>{}
                                )
  };

  BOOST_DISPATCH_IMPLEMENT( firstnonsingleton_, tag::cpu_
                          , (A0)(N)
                          , ((fusion_sequence_<A0,N>))
                          )
  {
    typedef std::size_t     result_type;

    BOOST_FORCEINLINE
    result_type operator()(A0 const& a0) const
    {
      for(std::size_t i = 0; i < a0.size(); ++i)
        if(a0[i] != 1) return i+1;

      return 1;
    }
  };
} }

#endif
