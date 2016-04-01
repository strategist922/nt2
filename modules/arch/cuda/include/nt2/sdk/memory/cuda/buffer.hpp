//==============================================================================
//         Copyright 20014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_MEMORY_CUDA_BUFFER_HPP_INCLUDED
#define NT2_SDK_MEMORY_CUDA_BUFFER_HPP_INCLUDED

#if defined(NT2_HAS_CUDA)

#include <nt2/sdk/memory/cuda/allocator.hpp>
#include <nt2/sdk/memory/cuda/pinned_allocator.hpp>
#include <nt2/include/functions/copy.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace nt2 { namespace memory
{
  template<class T>
  class cuda_buffer
  {
  public:
    //==========================================================================
    // Container types
    //==========================================================================

    typedef cuda_allocator<T>                               allocator_type;
    typedef typename allocator_type::value_type             value_type;
    typedef typename allocator_type::pointer                pointer;
    typedef typename allocator_type::const_pointer          const_pointer;
    typedef typename allocator_type::reference              reference;
    typedef typename allocator_type::const_reference        const_reference;
    typedef typename allocator_type::size_type              size_type;
    typedef typename allocator_type::difference_type        difference_type;
    typedef typename allocator_type::pointer                iterator;
    typedef typename allocator_type::const_pointer          const_iterator;

    //==========================================================================
    // Default constructor
    //==========================================================================
    cuda_buffer() : begin_(nullptr), end_(nullptr)
    {}

  public:
    //==========================================================================
    // Size constructor
    //==========================================================================
    cuda_buffer( size_type n)
          :  begin_(nullptr), end_(nullptr)
    {
      if(!n) return;
      allocator_type alloc_;

      begin_ = alloc_.allocate(n);
      end_ = begin_ + n;
    }

    //==========================================================================
    // Copy constructor
    //==========================================================================
    cuda_buffer( cuda_buffer const& src )
          : begin_(nullptr), end_(nullptr)
    {
      if(!src.size()) return;

      allocator_type alloc_;

      begin_ = alloc_.allocate(src.size());

      copy(src, *this);

      end_ = begin_ + src.size();
    }

    //==========================================================================
    // Destructor
    //==========================================================================
    ~cuda_buffer()
    {
      allocator_type alloc_;
      alloc_.deallocate(begin_);
    }

    //==========================================================================
    // Assignment
    //==========================================================================
    cuda_buffer& operator=(cuda_buffer const& src)
    {
      if(!src.size()) return *this;

      if( src.size() > this->size() )
      {
        allocator_type alloc_;
        alloc_.deallocate(begin_);
        begin_ = alloc_.allocate(src.size());
        end_ = begin_ + src.size();
      }

      copy(src, *this);

      return *this;
    }

    bool operator==(cuda_buffer const&)
    {
      return true;
    }

    template<typename container>
    bool operator==(container const&)
    {
      return false;
    }

    //==========================================================================
    // Swap
    //==========================================================================
    void swap( cuda_buffer& src )
    {
      std::swap(begin_          , src.begin_          );
      std::swap(end_            , src.end_            );
    }

    //==========================================================================
    // Resize
    //==========================================================================

    void resize( size_type sz )
    {
      if (sz < size())
      {
        end_ = begin_ + sz ;
      }
      else if (sz > size())
      {
       allocator_type alloc_;
       if (begin_ != nullptr)  alloc_.deallocate(begin_);
       begin_ = alloc_.allocate(sz);
       end_ = begin_ + sz;
      }
    }

    //==========================================================================
    // Iterators
    //==========================================================================
    iterator        begin()       { return begin_;  }
    const_iterator  begin() const { return begin_;  }
    iterator        end()         { return end_;    }
    const_iterator  end()   const { return end_;    }

    //==========================================================================
    // Raw values
    //==========================================================================
    pointer        data()       { return begin_;  }
    const_pointer  data() const { return begin_;  }

    //==========================================================================
    // Size related members
    //==========================================================================
    inline size_type  size()      const { return end_ - begin_;       }
    inline bool       empty()     const { return size() == 0;         }
    inline size_type  max_size()  const
    {
      return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }

    BOOST_FORCEINLINE reference operator[](size_type )
    {
      static constexpr value_type x = 0;
      static value_type result = 0;
      static_assert( x == 0 , "operator[] not available for cuda buffers");
      return result;
    }

    /// @overload
    BOOST_FORCEINLINE const_reference operator[](size_type ) const
    {
      static constexpr value_type x = 0 ;
      static_assert( x == 0 , "operator[] not available for cuda buffers");
      return x;
    }

  private:
    pointer     begin_, end_;

  };

  template<class T,class A> inline void swap(cuda_buffer<T>& x, cuda_buffer<T>& y)
  {
    x.swap(y);
  }

} }

#endif
#endif

