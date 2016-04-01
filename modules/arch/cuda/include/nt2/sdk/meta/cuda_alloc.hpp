//==============================================================================
//         Copyright 20014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_META_CUDA_ALLOC_HPP_INCLUDED
#define NT2_SDK_META_CUDA_ALLOC_HPP_INCLUDED

#if defined(NT2_HAS_CUDA)

#include <nt2/sdk/memory/cuda/pinned_allocator.hpp>

namespace nt2
  {
    template<typename Container>
    typename Container::value_type* cuda_get_mapped_device_pointer(Container & c )
    {
      typename Container::value_type * out = nullptr;
      if ( (cuda_alloc_type == cudaHostAllocMapped) && (locality(c) == pinned_ {}) ) cudaHostGetDevicePointer( (void **) &out, (void*) c.data() ,0 );
      return out;
    }
 }

#endif
#endif

