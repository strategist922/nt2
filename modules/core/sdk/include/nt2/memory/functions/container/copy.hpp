//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_MEMORY_FUNCTIONS_CONTAINER_COPY_HPP_INCLUDED
#define NT2_MEMORY_FUNCTIONS_CONTAINER_COPY_HPP_INCLUDED


namespace nt2 { namespace memory
{
  template<class T>
  class cuda_buffer;

  template<class T> inline void copy(cuda_buffer<T> const& a, cuda_buffer<T> & b )
  {
    b = a;
  }

} }

#endif
