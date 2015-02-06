//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_MEMORY_FUNCTIONS_CUDA_COPY_HPP_INCLUDED
#define NT2_MEMORY_FUNCTIONS_CUDA_COPY_HPP_INCLUDED

#if defined(NT2_HAS_CUDA)

#include <nt2/sdk/cuda/cuda.hpp>
#include <nt2/sdk/memory/cuda/buffer.hpp>
#include <cublas.h>

namespace nt2 { namespace memory
  {

  template<class T>
  class cuda_buffer;

  template<typename T, class C1>
  inline void copy(cuda_buffer<T> const& a, C1 &b, cudaStream_t stream = 0)
  {
    b.resize(a.size());

    CUDA_ERROR(cudaMemcpyAsync( b.data()
     , a.data()
     , a.size()* sizeof(T)
     , cudaMemcpyDeviceToHost
     , stream
     ));
  }

  template< class C1,typename T>
  inline void copy(C1 const& a, cuda_buffer<T> &b, cudaStream_t stream = 0)
  {
    b.resize(a.size());

    CUDA_ERROR(cudaMemcpyAsync( b.data()
     , a.data()
     , a.size()* sizeof(T)
     , cudaMemcpyHostToDevice
     , stream
     ));
  }

  template<typename T>
  inline void copy(cuda_buffer<T> const& a, cuda_buffer<T> & b, cudaStream_t stream = 0 )
  {
    b.resize(a.size());

    CUDA_ERROR(cudaMemcpyAsync( b.data()
     , a.data()
     , a.size()* sizeof(T)
     , cudaMemcpyDeviceToDevice
     , stream
     ));
  }
}}


#endif

#endif
