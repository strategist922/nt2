//==============================================================================
//         Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_FACTORY_OF_HPP_INCLUDED
#define BOOST_DISPATCH_META_FACTORY_OF_HPP_INCLUDED

#include <boost/dispatch/meta/model_of.hpp>
#include <boost/dispatch/meta/value_of.hpp>
#include <boost/dispatch/meta/primitive_of.hpp>
#include <boost/mpl/apply.hpp>

namespace boost { namespace dispatch { namespace details
{
  template<class T, class U>
  struct factory_of_impl
  {
    struct type
    {
      template<class X>
      struct apply
       : mpl::apply< typename meta::model_of<T>::type
                   , typename mpl::apply< typename factory_of_impl< typename meta::value_of<T>::type
                                                                  , U
                                                                  >::type
                                        , X
                                        >::type
                   >
      {
      };
    };
  };

  template<class T>
  struct factory_of_impl<T, T>
  {
    struct type
    {
      template<class X>
      struct apply
      {
        typedef X type;
      };
    };
  };
}

namespace meta
{
  /*!
    @brief Type factory generator

    For any Hierarchizable type, returns a @mpllambda which permits the lazy
    recomposition of any type of the same hierarchy from a set of types.

    @tparam Hierarchizable type to deconstruct

    @par Semantic:

    For any Hierarchizable @c H,

    @code
    typedef boost::dispatch::meta::factory_of<T>::type l;
    @endcode

    evaluates to a @mpllambda so that, for any types @c Tn...

    @code
    typedef boost::mpl::apply<l, Tn...>::type r;
    @endcode

    returns a type @c r which is a Hierarchizable of the same hierarchy than
    @c H and an equivalent semantic.

    @usage{factory_of.cpp}

  **/
  template<typename T, typename U = typename meta::primitive_of<T>::type>
  struct  factory_of
        : details::factory_of_impl<T, U>
  {
  };

} } }

#endif
