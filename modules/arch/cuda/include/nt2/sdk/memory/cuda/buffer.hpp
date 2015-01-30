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

#include <nt2/sdk/cuda/cuda.hpp>
#include <nt2/sdk/memory/cuda/allocator.hpp>
#include <boost/throw_exception.hpp>
#include <boost/assert.hpp>
#include <boost/swap.hpp>
#include <cublas.h>
#include <limits>

namespace nt2 { namespace memory
{
  template<class T>
  class cuda_buffer
  {
  public:
    //==========================================================================
    // Container types
    //==========================================================================

    typedef T                  value_type;
    typedef T*                 pointer;
    typedef T const*           const_pointer;
    typedef T&                 reference;
    typedef T const&           const_reference;
    typedef std::size_t        size_type;
    typedef std::ptrdiff_t     difference_type;
    typedef cuda_allocator<T>  allocator_type;
    typedef T*                 iterator;
    typedef T const*           const_iterator;

    //==========================================================================
    // Default constructor
    //==========================================================================
    cuda_buffer() : begin_(0), end_(0), stream_(0)
    {}

  public:
    //==========================================================================
    // Size constructor
    //==========================================================================
    cuda_buffer( size_type n)
          :  begin_(0), end_(0), stream_(0)
    {
      if(!n) return;

      CUDA_ERROR(cudaMalloc( reinterpret_cast<void**>(&begin_)
                           , n* sizeof(value_type)
                           ));
      end_ = begin_ + n;
    }

    //==========================================================================
    // Copy constructor
    //==========================================================================
    cuda_buffer( cuda_buffer const& src )
          : begin_(0), end_(0), stream_(0)
    {
      if(!src.size()) return;

      CUDA_ERROR(cudaMalloc( reinterpret_cast<void**>(&begin_)
                           , src.size()*sizeof(value_type)
                           ));

       CUDA_ERROR(cudaMemcpyAsync( begin_
                                 , src.data()
                                 , src.size()* sizeof(value_type)
                                 , cudaMemcpyDeviceToDevice
                                 , stream_
                                 ));

      end_ = begin_ + src.size();
    }

    template<typename Container>
    cuda_buffer( Container const& src )
          : begin_(0), end_(0), stream_(0)
    {
      if(!src.size()) return;

      CUDA_ERROR(cudaMalloc( reinterpret_cast<void**>(&begin_)
                           , src.size()*sizeof(value_type)
                           ));

      CUDA_ERROR(cudaMemcpyAsync( begin_
                                , src.data()
                                , src.size() * sizeof(value_type)
                                ,  cudaMemcpyHostToDevice
                                , stream_
                                ));
      end_ = begin_ + src.size();
    }

    //==========================================================================
    // Destructor
    //==========================================================================
    ~cuda_buffer()
    {
      if(stream_ != NULL)
      {
        cudaStreamDestroy(stream_);
      }

      if(begin_)
      {
        cudaFree(begin_);
      }
    }

    //==========================================================================
    // Assignment
    //==========================================================================
    cuda_buffer& operator=(cuda_buffer const& src)
    {
      if(!src.size()) return *this;

      if( src.size() > this->size() )
      {
        cudaFree(begin_);
        CUDA_ERROR(cudaMalloc( reinterpret_cast<void**>(&begin_)
                             , src.size()*sizeof(value_type)
                             ));

        end_ = begin_ + src.size();
      }

      CUDA_ERROR(cudaMemcpyAsync( begin_
                                , src.data()
                                , src.size()*sizeof(value_type)
                                , cudaMemcpyDeviceToDevice
                                , stream_
                                ));

      return *this;
    }

    template<class Container>
    cuda_buffer& operator=(Container const& src)
    {
      if(!src.size()) return *this;

      if( src.size() > this->size() )
      {
        cudaFree(begin_);
        CUDA_ERROR(cudaMalloc( reinterpret_cast<void**>(&begin_)
                             , src.size()*sizeof(value_type)
                             ));

        end_ = begin_ + src.size();
      }

      CUDA_ERROR(cudaMemcpyAsync( begin_
                                , src.data()
                                , src.size()* sizeof(value_type)
                                , cudaMemcpyHostToDevice
                                , stream_
                                ));

      return *this;
    }

    bool operator==(cuda_buffer const& src)
    {
      return true;
    }

    template<typename container>
    bool operator==(container const& src)
    {
      return false;
    }

    //==========================================================================
    // Swap
    //==========================================================================
    void swap( cuda_buffer& src )
    {
      boost::swap(begin_          , src.begin_          );
      boost::swap(end_            , src.end_            );
      boost::swap(stream_         , src.stream_         );
    }

    //==========================================================================
    // Resize
    //==========================================================================

    void resize( size_type sz )
    {
      if (sz < this->size())
      {
        end_ = begin_ + sz ;
      }
      else if (sz > this->size())
      {
       cudaFree(begin_);
       CUDA_ERROR(cudaMalloc( reinterpret_cast<void**>(&begin_)
                            , sz*sizeof(value_type)
                            ));

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

    template<typename Container>
    void data(Container & dst) const
    {
      if(!this->size()) return;
      if ( dst.size() != this->size() ) dst.resize(of_size(this->size(),1));

      CUDA_ERROR(cudaMemcpyAsync( dst.data()
                                , this->begin_
                                , this->size() * sizeof(value_type)
                                , cudaMemcpyDeviceToHost, stream_
                                ));
    }

    //==========================================================================
    // Size related members
    //==========================================================================
    inline size_type  size()      const { return end_ - begin_;       }
    inline bool       empty()     const { return size() == 0;         }
    inline size_type  max_size()  const
    {
      return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }

    //==========================================================================
    // Stream related -- if neccesary
    //==========================================================================
    inline cudaStream_t   stream()     const { return stream_;  }
    inline cudaError_t   setStream()     const
      { return cudaStreamCreate(&stream_); }


  private:
    pointer     begin_, end_;
    cudaStream_t stream_;
  };

  template<class T,class A> inline void swap(cuda_buffer<T>& x, cuda_buffer<T>& y)
  {
    x.swap(y);
  }

} }

#endif
#endif

