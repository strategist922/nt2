//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_CUDA_SETTINGS_SPECIFIC_CUDA_HPP_INCLUDED
#define NT2_SDK_CUDA_SETTINGS_SPECIFIC_CUDA_HPP_INCLUDED

#ifdef NT2_HAS_CUDA

#include <cuda_runtime.h>
#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <cstring>
#include <iostream>

#define CUDA_ERROR(status)                                                      \
BOOST_VERIFY_MSG( status == cudaSuccess, cudaGetErrorString(status))

namespace nt2{ namespace details
  {
    template<typename T>
    struct cu_buffers
    {
      std::vector<T*> host_pinned;
      std::vector<T*> device;
      std::size_t size;

      cu_buffers() : host_pinned(0), device(0) ,size(0) {}
      cu_buffers(std::size_t size_,std::size_t nstreams)
      {
        size = size_ ;
        allocate(size,nstreams);
      }

      void allocate(std::size_t size_,std::size_t nstreams)
      {
        size = size_;
        if(size != device.size())
        {
          std::size_t sizeof_ = size*sizeof(T);
          host_pinned.resize(nstreams);
          device.resize(nstreams);
          for(std::size_t i =0; i < nstreams ; ++i)
          {
            CUDA_ERROR(cudaMallocHost( (void**)&host_pinned[i] , sizeof_
                                ));

            CUDA_ERROR(cudaMalloc((void**)&device[i] , sizeof_  ));
          }
        }
      }

      T* get_host(std::size_t indx)
      {
        return host_pinned[indx];
      }

      T* get_device(std::size_t indx)
      {
        return device[indx];
      }

      template<class Container>
      void copy_hostpinned(Container & c,std::size_t streamid, std::size_t sizeb, int blockid)
      {
        std::memcpy(host_pinned[streamid], c.data() + blockid * size , sizeb*sizeof(T) );
      }


      template<class Container>
      void copy_host(Container & c, std::size_t streamid, std::size_t sizeb, int blockid)
      {
        std::memcpy(c.data() + blockid * size, host_pinned[streamid] , sizeb*sizeof(T) );
      }

      ~cu_buffers()
      {
        for(std::size_t i = 0 ; i < device.size() ; ++i )
        {
          CUDA_ERROR(cudaFreeHost(host_pinned[i]));
          CUDA_ERROR(cudaFree(device[i]));
        }
      }

    };

    template<typename Arch, typename T>
    struct specific_cuda
    {
      using btype = boost::dynamic_bitset<>;
      btype block_stream_dth;
      btype block_stream_htd;
      cu_buffers<T> buffers;
      std::size_t blocksize;
      bool allocated ;

      inline void synchronize() {}

      specific_cuda() : block_stream_dth(1), block_stream_htd(1) ,
                      buffers{} , blocksize(0) , allocated(false)
      {}

      ~specific_cuda()
      {
        block_stream_dth.reset();
        block_stream_htd.reset();
        allocated = false;
      }

      void swap(specific_cuda &)
      {

      }

      inline void allocate(std::size_t blocksize_ , std::size_t nstreams, std::size_t s)
      {
        if (!allocated)
        {
          blocksize = blocksize_ ;
          std::size_t num = s / blocksize_  ;
          block_stream_dth.resize(num);
          block_stream_htd.resize(num);
          buffers.allocate(blocksize, nstreams);
          allocated = true;
        }
      }

      template<class In, class Stream, class Set>
      inline void transfer_htd( In & in, int blockid, Stream & stream ,std::size_t streamid
                              , Set & addr, std::size_t leftover = 0)
      {
        std::size_t sizeb = blocksize;
        if(leftover !=0) sizeb = leftover ;

        auto it_set = addr.find(in.data());
        if( block_stream_htd[blockid] == false && (it_set == addr.end() ) )
        {
        block_stream_htd[blockid] = true;
        buffers.copy_hostpinned(in, streamid, sizeb , blockid);

        CUDA_ERROR(cudaMemcpyAsync( buffers.get_device(streamid)
                                  , buffers.get_host(streamid)
                                  , sizeb* sizeof(T)
                                  , cudaMemcpyHostToDevice
                                  , stream
                  ));

        }
      }

      template<class Out, class Stream>
      inline void transfer_dth( Out & out , int blockid, Stream & stream  ,std::size_t streamid
                              , std::size_t leftover = 0)
      {
        std::size_t sizeb = blocksize;
        if(leftover !=0) sizeb = leftover ;

        if(block_stream_dth[blockid] == false )
        {
          CUDA_ERROR(cudaMemcpyAsync( buffers.get_host(streamid)
                          , buffers.get_device(streamid)
                          , sizeb * sizeof(T)
                          , cudaMemcpyDeviceToHost
                          , stream
                    ));

          block_stream_dth[blockid] = true;
        }
      }

      inline T* data(std::size_t i)
      {
        return buffers.get_device(i);
      }

      template<typename Container>
      inline void copy_host(Container & c, std::size_t streamid,  std::size_t sizeb
                           ,  std::size_t blockid )
      {
        buffers.copy_host(c, streamid, sizeb , blockid);;
      }

    };

  }
}

#endif

#endif
