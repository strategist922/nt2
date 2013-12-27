//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_EXPONENTIAL_FUNCTIONS_FAST_LOG10_HPP_INCLUDED
#define NT2_EXPONENTIAL_FUNCTIONS_FAST_LOG10_HPP_INCLUDED

#include <nt2/include/functor.hpp>

namespace nt2 { namespace tag
  {
   /*!
     @brief fast_log10 generic tag

     Represents the fast_log10 function in generic contexts.

     @par Models:
        Hierarchy
   **/
    struct fast_log10_ : ext::elementwise_<fast_log10_>
    {
      /// @brief Parent hierarchy
      typedef ext::elementwise_<fast_log10_> parent;
    };
  }
  /*!
    base two logarithm function.

    @par Semantic:

    For every parameter of floating type T0

    @code
    T0 r = fast_log10(a0);
    @endcode

    is similar to:

    @code
    T0 r =  log(x)/log(Ten<T0>());;
    @endcode

    @param a0

    @see @funcref{log10}, @funcref{log}, @funcref{log1p}
    @return a value of the same type as the parameter
  **/
  NT2_FUNCTION_IMPLEMENTATION(tag::fast_log10_, fast_log10, 1)

}


#endif

