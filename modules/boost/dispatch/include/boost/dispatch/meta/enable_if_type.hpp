//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_ENABLE_IF_TYPE_HPP_INCLUDED
#define BOOST_DISPATCH_META_ENABLE_IF_TYPE_HPP_INCLUDED

namespace boost { namespace dispatch { namespace meta
{
  /*!
    @brief SFINAE-based type accessibility checker.

    Provides a SFINAE context to test the existence of a given type.

    @tparam T Type to check existence of.
    @tparam R Type to return if @c T is defined.

    @usage{enable_if_type.cpp}
  **/
  template<class T, class R=void>
  struct enable_if_type
  {
    typedef R type;
  };
} } }

#endif
