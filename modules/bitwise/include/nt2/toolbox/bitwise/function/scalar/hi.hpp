//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II         
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI         
//                                                                              
//          Distributed under the Boost Software License, Version 1.0.          
//                 See accompanying file LICENSE.txt or copy at                 
//                     http://www.boost.org/LICENSE_1_0.txt                     
//==============================================================================
#ifndef NT2_TOOLBOX_BITWISE_FUNCTION_SCALAR_HI_HPP_INCLUDED
#define NT2_TOOLBOX_BITWISE_FUNCTION_SCALAR_HI_HPP_INCLUDED

#include <nt2/sdk/meta/downgrade.hpp>
#include <nt2/sdk/meta/as_integer.hpp>

namespace nt2 { namespace meta
{
  NT2_FUNCTOR_IMPLEMENTATION( tag::hi_, tag::cpu_, (A0)
                            , (scalar_< arithmetic_<A0> >)
                            )

  {
    typedef typename meta::
            downgrade < typename meta::as_integer<A0,unsigned>::type
                      >::type  result_type;

    NT2_FUNCTOR_CALL(1)
    {
      typedef typename meta::as_integer<A0,unsigned>::type type;
      BOOST_STATIC_CONSTANT(type, shift   = sizeof(type)*4);
      BOOST_STATIC_CONSTANT(type, pattern = type(type(-1)<<shift));

      return b_and(pattern, a0) >> shift;
    }
  };
} }

#endif
