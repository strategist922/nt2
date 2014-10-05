//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_IS_RESULT_OF_SUPPORTED_HPP_INCLUDED
#define BOOST_DISPATCH_META_IS_RESULT_OF_SUPPORTED_HPP_INCLUDED

#include <boost/dispatch/meta/enable_if_type.hpp>
#include <boost/dispatch/meta/result_of.hpp>

namespace boost { namespace dispatch { namespace meta
{
  /*!
    @brief Check for result_of protocol support

    For a given type @c T describing a function type, this @metafunction
    verify that @c result_of<T>::type is well defined.


    @par Model:
    @metafunction

    @par Semantic:

    For any type @c T,

    @code
    typedef boost::dispatch::meta::is_result_of_supported<T>::type r;
    @endcode

    evaluates to @true_ if @c result_of<T>::type is defined.

    @usage{is_result_of_supported.cpp}

    @tparam T Function-type call to resolve

  **/
  template<typename T, typename Enable=void>
  struct  is_result_of_supported
#if !defined(DOXYGEN_ONLY)
        : boost::mpl::false_
#endif
  {};

  /// INTERNAL ONLY
  template<typename T>
  struct  is_result_of_supported<T
                                , typename
                                  enable_if_type<typename result_of<T>::type>::type
                                >
        : boost::mpl::true_
  {
  };
} } }

#endif
