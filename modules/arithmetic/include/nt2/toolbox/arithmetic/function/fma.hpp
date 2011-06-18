//////////////////////////////////////////////////////////////////////////////
///   Copyright 2003 and onward LASMEA UMR 6602 CNRS/U.B.P Clermont-Ferrand
///   Copyright 2009 and onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
///
///          Distributed under the Boost Software License, Version 1.0
///                 See accompanying file LICENSE.txt or copy at
///                     http://www.boost.org/LICENSE_1_0.txt
//////////////////////////////////////////////////////////////////////////////
#ifndef NT2_TOOLBOX_ARITHMETIC_FUNCTION_FMA_HPP_INCLUDED
#define NT2_TOOLBOX_ARITHMETIC_FUNCTION_FMA_HPP_INCLUDED
#include <nt2/include/simd.hpp>
#include <nt2/include/functor.hpp>
#include <nt2/toolbox/arithmetic/include.hpp>

namespace nt2 { namespace tag
  {         
    struct fma_ {};
  }
  NT2_FUNCTION_IMPLEMENTATION(tag::fma_, fma, 3)
  NT2_FUNCTION_IMPLEMENTATION(tag::fma_, madd, 3)
  
  NT2_FUNCTION_INTERFACE(tag::fma_, fam, 3)
  {
    typename make_functor<tag::fma_, A0>::type callee;
    return callee(a1, a2, a0);
  }
  
  NT2_FUNCTION_INTERFACE(tag::fma_, amul, 3)
  {
    typename make_functor<tag::fma_, A0>::type callee;
    return callee(a1, a2, a0);
  }
  
}
 
#include <nt2/toolbox/operator.hpp>
#include <nt2/toolbox/arithmetic/function/scalar/fma.hpp>
#include <nt2/toolbox/arithmetic/function/simd/all/fma.hpp> 

#include <nt2/toolbox/arithmetic/recognition/fma.hpp>
 
#endif

// modified by jt the 25/12/2010
