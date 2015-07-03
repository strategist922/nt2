#ifndef NT2_SDK_MEMORY_OPENCL_ALLOCATOR_HPP_INCLUDED
#define NT2_SDK_MEMORY_OPENCL_ALLOCATOR_HPP_INCLUDED

#if defined(NT2_HAS_OPENCL)

#include <nt2/sdk/opencl/opencl.hpp>

namespace nt2 { namespace memory
{
  template<typename T> struct opencl_allocator
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
      typedef opencl_allocator<U> other;
    };

    /// Default constructor
    opencl_allocator() {}

    /// Constructor from another opencl_allocator
    template<typename U> opencl_allocator(opencl_allocator<U> const& ) {}

    /// Constructor from another opencl_allocator
    template<typename U>
    opencl_allocator& operator=(opencl_allocator<U> const& )
    {
      return *this;
    }

    /// Destructor
    ~opencl_allocator() {}

    /// Retrieve the address of an element
    pointer       address(reference r)       { return &r; }

    /// @overload
    const_pointer address(const_reference r) { return &r; }

    /// Maximum amount of memory that can be allocated
    size_type max_size() const  { return size_type(~0); }

    /// Allocate a block of CUDA compatible memory
    pointer allocate( size_type c)
    {
      return NULL;
    }


    /// Deallocate a pointer allocated by the current opencl_allocator
    void deallocate (pointer p )
    {
    }
  };

  /// Equality comparison between two opencl_allocators
  template<typename T>
  bool operator== (opencl_allocator<T> const&, opencl_allocator<T> const&)
  {
    return true;
  }

  /// Inequality comparison between two opencl_allocators
  template<typename T>
  bool operator!= (opencl_allocator<T> const&, opencl_allocator<T> const&)
  {
    return false;
  }

} }

#endif
#endif
