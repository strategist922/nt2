//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_FORWARD_SHAPE_HPP_INCLUDED
#define NT2_CORE_SETTINGS_FORWARD_SHAPE_HPP_INCLUDED

#include <cstddef>

namespace nt2
{
  template< std::ptrdiff_t UpperBound
          , std::ptrdiff_t LowerBound
          >
  struct band_diagonal_;

  typedef band_diagonal_<-1,-1> general_;
  typedef band_diagonal_<-1, 0> upper_triangular_;
  typedef band_diagonal_< 0,-1> lower_triangular_;
  typedef band_diagonal_< 2, 2> pentadiagonal_;
  typedef band_diagonal_< 1, 1> tridiagonal_;
  typedef band_diagonal_< 1, 0> upper_bidiagonal_;
  typedef band_diagonal_< 0, 1> lower_bidiagonal_;
  typedef band_diagonal_< 0, 0> diagonal_;

  // struct positive_definite_   {};
  // struct uhess_               {};
  // struct symmetric_           {};

  namespace tag
  {
    /// @brief Option tag for shape options
    struct shape_;
  }
}

#endif
