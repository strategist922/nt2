//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_MATCH_OPTION_HPP_INCLUDED
#define NT2_CORE_SETTINGS_MATCH_OPTION_HPP_INCLUDED

#include <boost/mpl/bool.hpp>
#include <utility>

namespace nt2 { namespace meta
{
  /*!
   * @brief Check if a type is a valid Option
   *
   **/
  template<typename T, typename Option>
  struct  match_option
        : decltype( match_(std::declval<Option>(), std::declval<T>()) )
  {};

  /// INTERNAL ONLY
  template<typename Option>
  struct match_option<void, Option> : boost::mpl::false_
  {};

  /// INTERNAL ONLY
  template<typename T, typename Option>
  struct match_option<T*, Option> : match_option<T,Option>
  {};

  /// INTERNAL ONLY
  template<typename Option>
  struct lambda_match_option
  {
    template<typename T> struct apply : match_option<T,Option> {};
  };
} }

#endif
