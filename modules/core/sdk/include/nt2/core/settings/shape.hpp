//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_SHAPE_HPP_INCLUDED
#define NT2_CORE_SETTINGS_SHAPE_HPP_INCLUDED

#include <nt2/core/settings/option.hpp>
#include <nt2/core/settings/forward/shape.hpp>

namespace nt2 { namespace tag
{
  struct shape_
  {
    template<class T>
    struct apply : boost::mpl::false_
    {};

    typedef nt2::general_ default_type;
  };

  template<std::ptrdiff_t UpperBound, std::ptrdiff_t LowerBound>
  struct shape_::apply<nt2::band_diagonal_<UpperBound,LowerBound> >
                : boost::mpl::true_
  {};

  // template<>
  // struct shape_::apply<nt2::positive_definite_>
  //               : boost::mpl::true_
  // {};

  // template<>
  // struct shape_::apply<nt2::symmetric_>
  //               : boost::mpl::true_
  // {};
} }

#include <nt2/core/settings/details/shape.hpp>

#endif
