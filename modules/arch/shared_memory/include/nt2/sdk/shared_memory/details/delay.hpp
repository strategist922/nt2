//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_DELAY_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_DELAY_HPP_INCLUDED

#include <cstddef>

void foo(){}

namespace nt2 { namespace details
{

  inline void delay(std::size_t delaylength, float &)
  {
    for (std::size_t i = 0; i < delaylength; i++)
    {
       foo();
    }
  }

} }

#endif
