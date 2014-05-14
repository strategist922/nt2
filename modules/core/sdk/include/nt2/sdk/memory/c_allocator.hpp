//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_MEMORY_C_ALLOCATOR_HPP_INCLUDED
#define NT2_SDK_MEMORY_C_ALLOCATOR_HPP_INCLUDED

#include <cstdlib>
#include <cstddef>
#include <memory>

namespace nt2
{
  /*!
    @brief malloc/free based allocator

    c_allocator is standard allocator using malloc and free as its memory
    handling routine. This allow container allocated with it to be compatible
    with C-style memory handling.

    @tparam T Type of data to allocate
  **/
  template<typename T>
  struct c_allocator : public std::allocator<T>
  {
    /// INTERNAL ONLY
    typedef typename std::allocator<T>::pointer     pointer;

    /// INTERNAL ONLY
    typedef typename std::allocator<T>::size_type   size_type;

    /// INTERNAL ONLY
    typedef typename std::allocator<T>::value_type  value_type;

    /// rebind interface for c_allocator
    template<typename U> struct rebind
    {
      typedef c_allocator<U> other;
    };

    /// Default constructor
    c_allocator() {}

    /// Constructor from another c_allocator
    template<typename U> c_allocator(c_allocator<U> const&) {}

    /// Perform memory allocation
    pointer allocate( size_type s, const void* = 0 ) const
    {
      return static_cast<pointer>(::malloc(s*sizeof(value_type)));
    }

    /// Release allocated memory
    void deallocate(pointer p, size_type) const
    {
      ::free(p);
    }
  };

  /// c_allocator equality comparison
  template<typename T>
  bool operator==(c_allocator<T> const&, c_allocator<T> const&)
  {
    return true;
  }

  /// c_allocator inequality comparison
  template<typename T>
  bool operator!=(c_allocator<T> const& lhs, c_allocator<T> const& rhs)
  {
    return false;
  }
}

#endif
