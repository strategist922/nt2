//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_SIGN_OF_HPP_INCLUDED
#define BOOST_DISPATCH_META_SIGN_OF_HPP_INCLUDED

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/dispatch/meta/primitive_of.hpp>
#include <boost/type_traits/is_floating_point.hpp>

namespace boost { namespace dispatch { namespace meta
{
  template<typename T> struct sign_of;
} } }

#if !defined(DOXYGEN_ONLY)
namespace boost { namespace dispatch { namespace ext
{
  template<typename T, typename Enable = void>
  struct  sign_of
        : sign_of< typename meta::primitive_of<T>::type >
  {};

  template<typename T>
  struct sign_of<T, typename enable_if< boost::is_signed<T> >::type>
  {
    typedef signed type;
  };

  template<typename T>
  struct sign_of<T, typename enable_if< boost::is_unsigned<T> >::type>
  {
    typedef unsigned type;
  };

  template<typename T>
  struct sign_of<T, typename enable_if< boost::is_floating_point<T> >::type>
  {
    typedef signed type;
  };
} } }
#endif

namespace boost { namespace dispatch { namespace meta
{

  /*! Signedness of a type

    Computes the signedness of a type, ie the fact that the type is able to
    represent signed values like signed integers or floating point values.

    @par Semantic

    For any type @c T, the following code:

    @code
    typedef sign_of<T>::type r;
    @endcode

    is equivalent to :

    @code
    typedef signed r;
    @endcode

    if T is a type able to represent a signed value and to :

    @code
    typedef unsigned r;
    @endcode

    otherwise.

    @tparam T Type to analyze
   **/
  template<typename T>
  struct  sign_of
#if !defined(DOXYGEN_ONLY)
        : ext::sign_of<T>
#endif
  {};

  /// INTERNAL ONLY
  template<typename T> struct  sign_of<T&> : sign_of <T> {};

  /// INTERNAL ONLY
  template<typename T> struct  sign_of<T const> : sign_of <T> {};
} } }
#endif
