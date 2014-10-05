//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_COUNTER_HPP_INCLUDED
#define BOOST_DISPATCH_META_COUNTER_HPP_INCLUDED

#include <boost/dispatch/details/typeof.hpp>
#include <boost/preprocessor/cat.hpp>

#if defined(DOXYGEN_ONLY)
/*
  @brief Maximum value of static counter.

  If defined, this preprocessor symbol modify the maximum value of
  compile-time counters. If left unspecified, this value is set to 8.
*/
#define BOOST_DISPATCH_COUNTER_MAX
#else
#ifndef BOOST_DISPATCH_COUNTER_MAX
#define BOOST_DISPATCH_COUNTER_MAX 8
#endif
#endif

#if !defined(DOXYGEN_ONLY)
namespace boost { namespace dispatch { namespace details
{
  template<int N> struct depth_;

  template<>
  struct depth_<0>
  {
  };

  template<int N>
  struct depth_ : depth_<N-1>
  {
  };

  template<class T, class U = T>
  struct identity
  {
    typedef T type;
  };
}
#endif

namespace meta
{
  /// INTERNAL ONLY
  struct adl_helper_ {};
}
} }

/*
  @brief Static counter constructor

  Create a counter named @c NAME and initialize it to 0

  @param NAME Name of the static counter to initialize
*/
#define BOOST_DISPATCH_COUNTER_INIT(NAME)                                      \
namespace boost { namespace dispatch { namespace meta                          \
{                                                                              \
  boost::mpl::int_<0> NAME( details::depth_<0>, adl_helper_ const& );          \
} } }                                                                          \
/**/

/*
  @brief Static counter value accessor

  Retrieve the value of a static counter named @c

  @param NAME Name of the static counter to retrieve
*/
#define BOOST_DISPATCH_COUNTER_VALUE(NAME)                                     \
boost::dispatch::details::                                                     \
identity< BOOST_DISPATCH_TYPEOF(NAME                                           \
                                ( boost::dispatch::details                     \
                                       ::depth_<BOOST_DISPATCH_COUNTER_MAX>()  \
                                , boost::dispatch::meta::adl_helper_()         \
                                )                                              \
                               )                                               \
        >::type::value                                                         \
/**/

#define BOOST_DISPATCH_COUNTER_VALUE_TPL(name, dummy)                                              \
boost::dispatch::details::                                                                         \
identity< BOOST_DISPATCH_TYPEOF(name                                                               \
                                ( (typename boost::dispatch::details::                             \
                                   identity< boost::dispatch::details::                            \
                                             depth_<BOOST_DISPATCH_COUNTER_MAX>                    \
                                           , dummy                                                 \
                                           >::type()                                               \
                                  )                                                                \
                                , boost::dispatch::meta::adl_helper_()                             \
                                )                                                                  \
                               )                                                                   \
        >::type::value                                                                             \
/**/

#define BOOST_DISPATCH_COUNTER_INCREMENT(name)                                                     \
namespace boost { namespace dispatch { namespace meta                                              \
{                                                                                                  \
  boost::mpl::int_< BOOST_DISPATCH_COUNTER_VALUE(name) + 1 >                                       \
  name( details::depth_< BOOST_DISPATCH_COUNTER_VALUE(name) + 1 >                                  \
      , adl_helper_ const&                                                                         \
      );                                                                                           \
} } }                                                                                              \
/**/

#endif
