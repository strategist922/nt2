//==============================================================================
//         Copyright 2014 - 2015   NumScale
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_EXTERNAL_KERNEL_EXTERNAL_KERNEL_HPP
#define NT2_SDK_EXTERNAL_KERNEL_EXTERNAL_KERNEL_HPP

#include <boost/dispatch/details/parameters.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>

namespace nt2
{
  template<class Tag, class Site>
  struct external_kernel
  {
    #define M0(z, n, t)                                                        \
    template<BOOST_PP_ENUM_PARAMS(n, class A)>                                 \
    static void call(BOOST_PP_ENUM_BINARY_PARAMS(n, A, & a));                  \
    /**/
    BOOST_PP_REPEAT_FROM_TO(1, BOOST_PP_INC(BOOST_DISPATCH_MAX_ARITY), M0, ~)
    #undef M0
  };
}

#endif
