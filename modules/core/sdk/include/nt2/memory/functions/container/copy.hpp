//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_MEMORY_FUNCTIONS_CONTAINER_COPY_HPP_INCLUDED
#define NT2_MEMORY_FUNCTIONS_CONTAINER_COPY_HPP_INCLUDED

#include <nt2/sdk/memory/buffer.hpp>

namespace nt2 { namespace memory
{
  template<typename In, typename Out>
  inline void copy(In const& a, Out & b )
  {
    b = a;
  }

} }

#endif
