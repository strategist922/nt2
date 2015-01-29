//==============================================================================
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_MEMORY_BUFFER_HPP_INCLUDED
#define NT2_SDK_MEMORY_BUFFER_HPP_INCLUDED

#include <nt2/sdk/memory/adapted/buffer.hpp>
#include <nt2/sdk/memory/details/buffer.hpp>
#include <nt2/sdk/memory/copy.hpp>
#include <initializer_list>
#include <type_traits>
#include <cstddef>
#include <memory>

namespace nt2 { namespace memory
{
  /*!
    @brief Releasable sequence container

    buffer is a sequence container that encapsulates dynamic size arrays with
    various enhancement with respect to memory ownership and reuse.

    Main difference with std::vector are:

      * a release() method which, similarly to unique_ptr, allow to extract data
        from a buffer
      * a reuse() method with perform destructive resizing.
      * the non-implementation of some members or members variation
        for KISS reasons.

    @tparam T The type of the elements.
    @tparam A An allocator that is used to acquire memory to store the elements.
              The type must meet the requirements of Allocator.
  **/
  template<typename T, typename A> struct buffer
  {
    /// INTERNAL ONLY
    using alloc_t         = std::allocator_traits<A>;

    /// INTERNAL ONLY
    using del_t           = details::deleter<A>;

    /// INTERNAL ONLY
    using data_type       = std::unique_ptr<T[],del_t>;

    using allocator_type  = typename alloc_t::allocator_type;
    using value_type      = typename alloc_t::value_type;
    using iterator        = typename alloc_t::pointer;
    using const_iterator  = typename alloc_t::const_pointer;
    using pointer         = typename alloc_t::pointer;
    using const_pointer   = typename alloc_t::const_pointer;
    using size_type       = typename alloc_t::size_type;
    using difference_type = typename alloc_t::difference_type;
    using reference       = typename std::iterator_traits<iterator>::reference;
    using const_reference = typename std::iterator_traits<const_iterator>::reference;

    /*!
      @brief Constructs a buffer of n elements

      @param n Number of elements to allocate
      @param a Allocator value to use as initialization
    **/
    buffer( std::size_t n, A const& a )
          : alloc_{a}
          , data_{alloc_t::allocate(alloc_,n), del_t{&alloc_,n,n}}
          , size_{n}, capacity_{n}
    {
      details::may_construct( begin() , end(), alloc_ );
    }

    /*!
      @brief Default Constructor with allocator support

      Constructs an empty buffer and initializes its allocator

      @param a Allocator value to use as initialization
    **/
    buffer(A const& a)
          : alloc_{a}, data_{pointer{0}, del_t{&alloc_,0,0}}
          , size_{0}, capacity_{0}
    {}


    /*!
      @brief Default Constructor

      Constructs an empty buffer.
    **/
    BOOST_FORCEINLINE  buffer() : buffer(A{}) {}

    /*!
      @brief Constructs a buffer of n elements

      @param n Number of elements to allocate
    **/
    buffer( std::size_t n ) : buffer{n,A{}} {}

    /// @brief Copy Constructor
    buffer( buffer const& other ) : buffer(other.size_,other.alloc_)
    {
      nt2::memory::copy(other.begin(),other.end(),begin());
    }

    /// @brief Move constructor
    BOOST_FORCEINLINE buffer(buffer&& other) : buffer() { swap( other ); }

    /// @brief Initializer list constructor
    buffer(std::initializer_list<T> args) : buffer(args.size())
    {
      nt2::memory::copy(args.begin(),args.end(),begin());
    }

    /*!
      @brief Copy assignment operator

      Replaces the contents with a copy of the contents of other.

      @param  other Data to copy
      @return Current instance with copied data
    **/
    buffer& operator=(buffer const& other)
    {
      if(&other != this)
      {
        buffer that(other);
        swap(that);
      }

      return *this;
    }

    /*!
      @brief Copy assignment operator from r-value reference

      Replaces the contents with the contents of other.

      @param other Data to extract data from
      @return Current instance with assigned data
    **/
    BOOST_FORCEINLINE buffer& operator=(buffer&& other)
    {
      swap(other);
      return *this;
    }

    /*!
      @brief Copy assignment operator from initializer list

      Replaces the contents with those identified by the initializer list.

      @param args List of value to copy from
      @return Current instance with copied data
    **/
    buffer& operator=(std::initializer_list<T> args)
    {
      buffer that(args);
      swap(that);

      return *this;
    }

    /// @brief Check if the buffer contains 0 element
    BOOST_FORCEINLINE bool  empty() const { return !size_;  }

    /// @brief Return the logical number of elements of the buffer
    BOOST_FORCEINLINE std::size_t size()  const { return size_; }

    /// @brief Return the allocated number of elements of the buffer
    BOOST_FORCEINLINE std::size_t capacity()  const { return capacity_; }

    /*!
      @brief Access to a specified element

      Returns a reference to the element at specified location i.
      No bounds checking is performed.

      @param i position of the element to return

      @return Reference to the requested element.
    **/
    BOOST_FORCEINLINE T& operator[](std::size_t i)
    {
      return data_[i];
    }

    /// @overload
    BOOST_FORCEINLINE T const& operator[](std::size_t i) const
    {
      return data_[i];
    }

    /*!
      @brief Exchanges the given buffers' values

      @params other value to be swapped
    **/
    void swap(buffer& other)
    {
      using std::swap;
      swap( alloc_   , other.alloc_    );
      swap( data_    , other.data_     );
      swap( size_    , other.size_     );
      swap( capacity_, other.capacity_ );
    }

    /*!
      @brief Changes the number of elements stored

      Resizes the container to contain @c n elements.

      @param n  new size of the buffer
    **/
    void resize( std::size_t n )
    {
      realloc ( n , [](buffer const& c, pointer o)
                    {
                      nt2::memory::copy( c.begin(),c.end(), o );
                    }
                  , [&](std::size_t sz) { complete_realloc(sz); }
              );
    }

    /*!
      @brief Reset the number of elements stored

      Resizes the container to contain @c n elements while losing any
      preexistign data.

      @param n  new size of the buffer
    **/
    void reuse( std::size_t n )
    {
      realloc ( n , [](buffer const&, pointer)  {}
                  , [&](std::size_t sz) { complete_realloc(sz); }
              );
    }

    /*!
      @brief Adds element to the end

      Appends the given element value to the end of the buffer.

      @param v element to insert
    **/
    void push_back(const_reference v)
    {
      auto pos        = size_;
      auto next_size  = size_+1;

      realloc ( next_size
              , [](buffer const& c, pointer o)
                {
                  nt2::memory::copy( c.begin(),c.end(), o );
                }
              , [&](std::size_t) { alloc_t::construct(alloc_, &data_[pos], v); }
              );
    }

    /*!
      @brief Adds elements to the end

      Appends the each elements of the given range to the end of the buffer.

      @param b begining of the range to insert
      @param e end of the range to insert

    **/
    template<typename Iterator> void append( Iterator b, Iterator e )
    {
      auto s = size_;
      resize(s + (e-b));
      nt2::memory::copy(b,e,begin()+s);
    }

    /*!
      @brief Release buffer data ownership

      Returns a pointer to the buffer data and releases the ownership.

      @return Pointer to the buffer data.
    **/
    BOOST_FORCEINLINE T* release()
    {
      size_ = capacity_ = 0;
      return data_.release();
    }

    /// @brief Returns the allocator associated with the container.
    BOOST_FORCEINLINE allocator_type& get_allocator()
    {
      return alloc_;
    }

    /// @overload
    BOOST_FORCEINLINE allocator_type const& get_allocator() const
    {
      return alloc_;
    }

    /// @brief Direct access to the underlying array
    BOOST_FORCEINLINE pointer  data()  { return data_.get(); }

    /// @overload
    BOOST_FORCEINLINE const_pointer  data() const  { return data_.get(); }

    /// @brief Iterator referencing the first buffer element
    BOOST_FORCEINLINE iterator begin() { return data_.get(); }

    /// @overload
    BOOST_FORCEINLINE const_iterator begin() const { return data_.get(); }

    /// @brief Iterator referencing the after-the-last buffer element
    BOOST_FORCEINLINE iterator end() { return data_.get() + size_;  }

    /// @overload
    BOOST_FORCEINLINE const_iterator end() const { return data_.get() + size_; }

    private:

    /// INTERNAL ONLY - Helper function that finish the work of resize/reuse
    void complete_realloc(std::size_t n)
    {
      if(n < size_)
        details::may_destroy( begin() + n, end(), alloc_ );
      else
        details::may_construct( end(), begin() + n, alloc_ );
    }

    /// INTERNAL ONLY - Helper function that execute a customizable reallocation
    template<typename Pre, typename Post>
    void realloc(std::size_t n, Pre pre_process, Post post_process)
    {
      if(n > capacity_)
      {
        // x1.5 is better than x2 as it maximizes page reuse
        capacity_ = 1 + n + n/2;

        data_type local { alloc_t::allocate(alloc_,capacity_)
                        , del_t{&alloc_,n,capacity_}
                        };

        details::may_construct( local.get(), local.get()+size_, alloc_ );
        pre_process(*this,local.get());
        data_.swap(local);
      }
      else
      {
        // Update deleter size info for correct deallocation & destruction
        data_.get_deleter().sz = n;
      }

      post_process(n);
      size_ = n;
    }

    A           alloc_;
    data_type   data_;
    std::size_t size_, capacity_;
  };

  /*!
    @brief Exchanges the given buffers' values

    @params a buffer to be swapped
    @params b buffer to be swapped
  **/
  template<typename T, typename A>
  BOOST_FORCEINLINE void swap( buffer<T,A>& a, buffer<T,A>& b )
  {
    a.swap(b);
  }
} }

#endif
