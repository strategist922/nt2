//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_ARITHMETIC_FUNCTIONS_DIVS_HPP_INCLUDED
#define NT2_ARITHMETIC_FUNCTIONS_DIVS_HPP_INCLUDED

#include <boost/simd/arithmetic/functions/divs.hpp>

namespace nt2
{
  namespace tag
  {
    using boost::simd::tag::divs_;
  }

  using boost::simd::divs;
  using boost::simd::saturated_div;
  using boost::simd::rdivide;
  using boost::simd::ldivide;
  using boost::simd::ldiv;
}

#endif
