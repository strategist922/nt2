//==============================================================================
//         Copyright 20014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_MEMORY_CUDA_PINNED_ALLOCATOR_HPP_INCLUDED
#define NT2_SDK_MEMORY_CUDA_PINNED_ALLOCATOR_HPP_INCLUDED

#if defined(NT2_HAS_CUDA)

#include <nt2/sdk/cuda/cuda.hpp>
#include <nt2/core/settings/locality.hpp>
#include <nt2/sdk/memory/is_compatible_allocator.hpp>
#include <cuda_runtime.h>
#include <type_traits>

#if defined(NT2_CUDA_INTEGRATED)
#define cuda_alloc_type cudaHostAllocMapped
#else
#define cuda_alloc_type cudaHostAllocDefault
#endif

namespace nt2 { namespace memory
{
#if defined(NT2_CUDA_INTEGRATED)
struct integrated_cuda_settings
{
  integrated_cuda_settings() { cudaSetDeviceFlags(cudaDeviceMapHost); }
};

static const integrated_cuda_settings proper_settings= {};
#endif


  template<typename T> struct cuda_pinned_
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
      typedef cuda_pinned_<U> other;
    };

    template<typename Container> struct traits
    {
      using k_t  = typename Container::kind_type;
      using s_t  = typename Container::settings_type;
      using ss_t = typename Container::scheme_t;
      using buffer_type = typename ss_t::template apply<Container>::type;
    };

    /// Default constructor
    cuda_pinned_() {}

    /// Constructor from another cuda_pinned_
    template<typename U> cuda_pinned_(cuda_pinned_<U> const& ) {}

    /// Constructor from another cuda_pinned_
    template<typename U>
    cuda_pinned_& operator=(cuda_pinned_<U> const& )
    {
      return *this;
    }

    /// Destructor
    ~cuda_pinned_() {}

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
      CUDA_ERROR(cudaHostAlloc( reinterpret_cast<void**>(&ptr)
                                     , c*sizeof(value_type)
                                     ,cuda_alloc_type
                           ));
      return ptr;
    }


    /// Deallocate a pointer allocated by the current cuda_pinned_
    void deallocate (pointer p , std::size_t )
    {
      if ( p != nullptr ) CUDA_ERROR(cudaFreeHost(p));
    }
  };

  /// Equality comparison between two cuda_pinned_s
  template<typename T>
  bool operator== (cuda_pinned_<T> const&, cuda_pinned_<T> const&)
  {
    return true;
  }

  /// Inequality comparison between two cuda_pinned_s
  template<typename T>
  bool operator!= (cuda_pinned_<T> const&, cuda_pinned_<T> const&)
  {
    return false;
  }

  template<typename RefAlloc, typename T>
  struct  is_compatible_allocator<RefAlloc, nt2::memory::cuda_pinned_<T>>
        : std::is_same<T, typename RefAlloc::value_type>
  {};

  template<typename OtherAlloc, typename T>
  struct  is_compatible_allocator<nt2::memory::cuda_pinned_<T>,OtherAlloc>
        : std::is_same<T, typename OtherAlloc::value_type>
  {};

  template<typename T, typename U>
  struct  is_compatible_allocator < nt2::memory::cuda_pinned_<T>
                                  , nt2::memory::cuda_pinned_<U>
                                  >
        : std::is_same<T, U>
  {};
}

using pinned_ = nt2::memory::cuda_pinned_<char>;

}

#endif
#endif

