//////////////////////////////////////////////////////////////////////////////
///   Copyright 2003 and onward LASMEA UMR 6602 CNRS/U.B.P Clermont-Ferrand
///   Copyright 2009 and onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
///
///          Distributed under the Boost Software License, Version 1.0
///                 See accompanying file LICENSE.txt or copy at
///                     http://www.boost.org/LICENSE_1_0.txt
//////////////////////////////////////////////////////////////////////////////
#ifndef NT2_TOOLBOX_CRLIBM_FUNCTION_SINH_RZ_HPP_INCLUDED
#define NT2_TOOLBOX_CRLIBM_FUNCTION_SINH_RZ_HPP_INCLUDED
#include <nt2/include/simd.hpp>
#include <nt2/include/functor.hpp>
#include <nt2/toolbox/crlibm/include.hpp>

namespace nt2 { namespace crlibm { namespace tag
  {         
    struct sinh_rz_ {};
  }
  NT2_FUNCTION_IMPLEMENTATION(crlibm::tag::sinh_rz_, sinh_rz, 1)
  } }
 
#include <nt2/toolbox/crlibm/function/scalar/sinh_rz.hpp>
// #include NT2_CRLIBM_INCLUDE(sinh_rz.hpp) 

 
#endif

// modified by jt the 29/12/2010