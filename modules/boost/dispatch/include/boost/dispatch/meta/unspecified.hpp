//==============================================================================
//         Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_UNSPECIFIED_HPP_INCLUDED
#define BOOST_DISPATCH_META_UNSPECIFIED_HPP_INCLUDED

#include <boost/dispatch/meta/details/hierarchy_pp.hpp>
#include <boost/dispatch/meta/unknown.hpp>

namespace boost { namespace dispatch { namespace meta
{
  /*!
    @brief Hierarchy for unregistered type

    The unspecified_ hierarchy is used for non-categorized type.

    @par Model:

    Hierarchy

    @tparam Type Type to be cateorized
  **/
  template<typename Type> struct unspecified_ : unknown_<Type>
  {
    /// @brief Parent Hierarchy
    typedef unknown_<Type>  parent;
  };
} } }

#endif
