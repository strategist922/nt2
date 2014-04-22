//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef FILTERS_BINARY_EROSION_HPP_INCLUDED
#define FILTERS_BINARY_EROSION_HPP_INCLUDED

#include <cstddef>
#include <boost/simd/include/functions/simd/bitwise_or.hpp>
#include <boost/dispatch/attributes.hpp>
#include "stencil.hpp"

namespace nt2
{
  // binary_erosion is a compact stencil computing the minimum
  // of its structuring element's content
  template<std::size_t Height, std::size_t Width>
  struct binary_erosion_ : compact_
  {
    static const std::size_t height = Height;
    static const std::size_t width  = Width;

    typedef tag::reductible_stencil_ filter_tag;

    template<std::size_t H, std::size_t W>
    struct rebind
    {
      typedef binary_erosion_<H,W> other;
    };

    template<typename T>
    BOOST_FORCEINLINE static T call(T const& value, T const& state)
    {
      return value | state;
    }
  };

  // naive_binary_erosion is an binary_erosion with optimisation disabled
  // DEMO PURPOSE ONLY
  template<std::size_t Height, std::size_t Width>
  struct naive_binary_erosion_ : binary_erosion_<Height,Width>
  {
    typedef tag::regular_stencil_ filter_tag;
  };
}

#endif
