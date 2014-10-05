//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_DSL_DETAILS_EXPR_OF_HPP_INCLUDED
#define BOOST_DISPATCH_DSL_DETAILS_EXPR_OF_HPP_INCLUDED

#include <boost/dispatch/meta/value_of.hpp>

namespace boost { namespace dispatch { namespace details
{
  template<typename T, typename Enable = void>
  struct expr_of
       : expr_of<typename meta::value_of<T>::type>
  {
  };

  template<typename T>
  struct expr_of<T, typename T::proto_is_expr_>
  {
    typedef T type;
  };
} } }

#endif
