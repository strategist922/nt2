//==============================================================================
//         Copyright 20014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_MEMORY_CUDA_ALLOCATOR_HPP_INCLUDED
#define NT2_SDK_MEMORY_CUDA_ALLOCATOR_HPP_INCLUDED

#if defined(NT2_HAS_CUDA)

#include <nt2/sdk/cuda/cuda.hpp>
#include <cuda_runtime.h>

namespace nt2 { namespace memory
{
  template<typename T> struct cuda_allocator
  {
    typedef T               value_type;
    typedef T*              pointer;
    typedef T const*        const_pointer;
    typedef T&              reference;
    typedef T const&        const_reference;
    typedef std::size_t     size_type;
    typedef std::ptrdiff_t  difference_type;

    template<typename U> struct rebind
    {
      typedef cuda_allocator<U> other;
    };

    /// Default constructor
    cuda_allocator() {}

    /// Constructor from another cuda_allocator
    template<typename U> cuda_allocator(cuda_allocator<U> const& ) {}

    /// Constructor from another cuda_allocator
    template<typename U>
    cuda_allocator& operator=(cuda_allocator<U> const& )
    {
      return *this;
    }

    /// Destructor
    ~cuda_allocator() {}

    /// Retrieve the address of an element
    pointer       address(reference r)       { return &r; }

    /// @overload
    const_pointer address(const_reference r) { return &r; }

    /// Maximum amount of memory that can be allocated
    size_type max_size() const  { return size_type(~0); }

    /// Allocate a block of CUDA compatible memory
    pointer allocate( size_type c)
    {
      pointer ptr;
      auto err = cudaMalloc( reinterpret_cast<void**>(&ptr)
                           , c*sizeof(value_type)
                           );

      return ptr;
    }


    /// Deallocate a pointer allocated by the current cuda_allocator
    void deallocate (pointer p )
    {
      if ( p != nullptr ) auto err = cudaFree(p);
    }
  };

  /// Equality comparison between two cuda_allocators
  template<typename T>
  bool operator== (cuda_allocator<T> const&, cuda_allocator<T> const&)
  {
    return true;
  }

  /// Inequality comparison between two cuda_allocators
  template<typename T>
  bool operator!= (cuda_allocator<T> const&, cuda_allocator<T> const&)
  {
    return false;
  }

} }

#endif
#endif

