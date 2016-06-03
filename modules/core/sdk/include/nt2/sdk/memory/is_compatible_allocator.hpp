//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_MEMORY_IS_COMPATIBLE_ALLOCATOR_HPP_INCLUDED
#define NT2_SDK_MEMORY_IS_COMPATIBLE_ALLOCATOR_HPP_INCLUDED

#include <type_traits>

namespace nt2 { namespace memory
{
  template<typename RefAlloc, typename OtherAlloc>
  struct  is_compatible_allocator
        : std::is_same<RefAlloc,OtherAlloc>
  {};
} }

#endif
