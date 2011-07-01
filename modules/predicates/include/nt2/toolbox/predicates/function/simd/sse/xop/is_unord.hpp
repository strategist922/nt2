//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II         
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI         
//                                                                              
//          Distributed under the Boost Software License, Version 1.0.          
//                 See accompanying file LICENSE.txt or copy at                 
//                     http://www.boost.org/LICENSE_1_0.txt                     
//==============================================================================
#ifndef NT2_TOOLBOX_PREDICATES_FUNCTION_SIMD_SSE_XOP_IS_UNORD_HPP_INCLUDED
#define NT2_TOOLBOX_PREDICATES_FUNCTION_SIMD_SSE_XOP_IS_UNORD_HPP_INCLUDED
#include <nt2/include/functions/boolean.hpp>
#include <nt2/sdk/details/ignore_unused.hpp>
#include <nt2/sdk/meta/strip.hpp>
/////////////////////////////////////////////////////////////////////////////
// Implementation when type A0 is arithmetic_
/////////////////////////////////////////////////////////////////////////////
namespace nt2 { namespace meta
{
  NT2_FUNCTOR_IMPLEMENTATION(tag::is_unord_, tag::cpu_,
                           (A0),
                           ((simd_<arithmetic_<A0>,tag::xop_>))
                           ((simd_<arithmetic_<A0>,tag::xop_>))
                          )
  {
    typedef A0 result_type;
    NT2_FUNCTOR_CALL_REPEAT(2)
    {
      ignore_unused(a0);
      ignore_unused(a1);
      return False<A0>();
    }
  };

/////////////////////////////////////////////////////////////////////////////
// Implementation when type A0 is double
/////////////////////////////////////////////////////////////////////////////


  NT2_FUNCTOR_IMPLEMENTATION(tag::is_unord_, tag::cpu_,
                           (A0),
                           ((simd_<double_<A0>,tag::xop_>))
                           ((simd_<double_<A0>,tag::xop_>))
                          )
  {
    typedef A0 result_type;
    NT2_FUNCTOR_CALL_REPEAT(2)
    {
      A0 that = {_mm256_cmp_pd(a0,a1, _CMP_UNORD_Q)};
      return that;
    }
  };

/////////////////////////////////////////////////////////////////////////////
// Implementation when type A0 is float
/////////////////////////////////////////////////////////////////////////////


  NT2_FUNCTOR_IMPLEMENTATION(tag::is_unord_, tag::cpu_,
                           (A0),
                           ((simd_<float_<A0>,tag::xop_>))
                           ((simd_<float_<A0>,tag::xop_>))
                          )
  {
    typedef A0 result_type;
    NT2_FUNCTOR_CALL_REPEAT(2)
    {
      A0 that = {_mm256_cmp_ps(a0,a1, _CMP_UNORD_Q)};
      return that;
    }
  };
} }
#endif
