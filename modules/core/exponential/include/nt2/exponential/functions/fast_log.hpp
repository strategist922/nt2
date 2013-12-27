//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_EXPONENTIAL_FUNCTIONS_FAST_LOG_HPP_INCLUDED
#define NT2_EXPONENTIAL_FUNCTIONS_FAST_LOG_HPP_INCLUDED
#include <nt2/include/functor.hpp>


namespace nt2 { namespace tag
  {
   /*!
     @brief fast_log generic tag

     Represents the fast_log function in generic contexts.

     @par Models:
        Hierarchy
   **/
    struct fast_log_ : ext::elementwise_<fast_log_>
    {
      /// @brief Parent hierarchy
      typedef ext::elementwise_<fast_log_> parent;
    };
  }
  /*!
    Natural fast_logarithm function.

    @par Semantic:

    For every parameter of floating type T0

    @code
    T0 r = fast_log(x);
    @endcode

    @see @funcref{fast_log10}, @funcref{fast_log2}, @funcref{fast_log1p}

    @param a0

    @return a value of the same type as the parameter
  **/
  NT2_FUNCTION_IMPLEMENTATION(tag::fast_log_, fast_log, 1)
}

#endif

