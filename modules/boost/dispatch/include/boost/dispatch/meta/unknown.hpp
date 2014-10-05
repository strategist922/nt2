//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2011 - 2013 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_UNKNOWN_HPP_INCLUDED
#define BOOST_DISPATCH_META_UNKNOWN_HPP_INCLUDED

namespace boost { namespace dispatch { namespace meta
{
  /*!
    @brief Hierarchy for unknown type

    The unknown_ hierarchy is the upper bound in the hierarchy lattice.
    When a dispatch resolves on unknown_, it means no suitable overload has
    been found.

    @par Model:

    Hierarchy

    @tparam Type Type to be categorized
  **/

  template<typename Type> struct unknown_
  {
    /// INTERNAL ONLY
    typedef Type      type;

    /// INTERNAL ONLY
    typedef unknown_  parent;

    /// INTERNAL ONLY
    typedef void      hierarchy_tag;
  };
} } }

#endif
