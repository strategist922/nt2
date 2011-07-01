//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II         
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI         
//                                                                              
//          Distributed under the Boost Software License, Version 1.0.          
//                 See accompanying file LICENSE.txt or copy at                 
//                     http://www.boost.org/LICENSE_1_0.txt                     
//==============================================================================
#ifndef NT2_TOOLBOX_IEEE_FUNCTION_SIMD_COMMON_MANTISSA_HPP_INCLUDED
#define NT2_TOOLBOX_IEEE_FUNCTION_SIMD_COMMON_MANTISSA_HPP_INCLUDED
#include <nt2/sdk/meta/adapted_traits.hpp>
#include <nt2/include/constants/properties.hpp>
#include <nt2/sdk/meta/as_integer.hpp>
#include <nt2/sdk/meta/strip.hpp>
#include <nt2/include/functions/is_eqz.hpp>
#include <nt2/include/functions/select.hpp>
/////////////////////////////////////////////////////////////////////////////
// Implementation when type  is arithmetic_
/////////////////////////////////////////////////////////////////////////////
namespace nt2 { namespace meta
{
  NT2_FUNCTOR_IMPLEMENTATION(tag::mantissa_, tag::cpu_,
                           (A0)(X),
                           ((simd_<arithmetic_<A0>,X>))
                          )
  {
    typedef A0 result_type;
    NT2_FUNCTOR_CALL_REPEAT(1)
    {
      typedef typename meta::as_integer<A0, unsigned>::type  int_type;
      typedef typename meta::scalar_of<int_type>::type      sint_type;
      typedef typename meta::scalar_of<A0>::type               s_type;
      const sint_type n1 = ((2*Maxexponent<s_type>()+1)<<Nbmantissabits<s_type>());
      const sint_type n2 = (sizeof(sint_type)-2);
      const int_type  mask0 = (splat<int_type>((n1<<2)>>2));
      const int_type  mask1 = (splat<int_type>((~n1)|n2));
      return sel(b_or(is_invalid(a0),is_eqz(a0)),a0,b_or(b_and(a0,mask1),mask0));
    }
  };
} }
#endif
