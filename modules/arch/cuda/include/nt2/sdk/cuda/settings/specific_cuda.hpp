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

#include <nt2/sdk/cuda/cuda.hpp>
#include <nt2/sdk/meta/locality.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <cstring>

namespace nt2
{
  namespace memory
  {
    template<typename T>
    class cuda_pinned_;
  }
  using pinned_ = nt2::memory::cuda_pinned_<char>;
}

namespace nt2{ namespace details
  {
    template<typename T>
    struct cu_buffers
    {
      std::size_t size;
      std::vector<T*> device ;
      std::vector<T*> host_pinned;
      std::size_t ns;

      cu_buffers() :size(0), ns(0)
      {
      }

      cu_buffers(std::size_t size_,std::size_t nstreams) : size(0), ns(0)
      {
        allocate(size,nstreams);
      }

      void allocate(std::size_t size_,std::size_t nstreams , nt2::host_ &)
      {
        if(size_ > size)
        {
          if(size != 0)
          {
            for(std::size_t i =0; i < device.size(); ++i)
            {
              CUDA_ERROR(cudaFreeHost(host_pinned[i]));
              CUDA_ERROR(cudaFree(device[i]));
            }
          }
          ns = nstreams;
          size = size_;
          std::size_t sizeof_ = size*sizeof(T);
          host_pinned.resize(nstreams);
          device.resize(nstreams);
          for(std::size_t i =0; i < nstreams; ++i)
          {
            CUDA_ERROR(cudaMallocHost( (void**)&host_pinned[i] , sizeof_ ));
            CUDA_ERROR(cudaMalloc((void**)&device[i] , sizeof_  ));

          }
        }
      }

      void allocate(std::size_t size_,std::size_t nstreams , nt2::pinned_ &)
      {
        if(size_ > size)
        {
          if(size != 0)
          {
            for(std::size_t i =0; i < device.size(); ++i)
            {
              CUDA_ERROR(cudaFree(device[i]));
            }
          }
          ns = nstreams;
          size = size_;
          std::size_t sizeof_ = size*sizeof(T);
          host_pinned.resize(0);
          device.resize(nstreams);
          for(std::size_t i =0; i < nstreams; ++i)
          {
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
          CUDA_ERROR(cudaFree(device[i]));
        }

        for(std::size_t i = 0 ; i < host_pinned.size() ; ++i )
        {
          CUDA_ERROR(cudaFreeHost(host_pinned[i]));
        }

        size = 0;
        device.resize(0);
        host_pinned.resize(0);
      }

    };

    template<typename Arch, typename T>
    struct specific_cuda
    {
      using btype = std::vector<bool>;
      btype block_stream_dth;
      btype block_stream_htd;
      cu_buffers<T> buffers;
      std::size_t blocksize;
      bool allocated ;

      inline void synchronize() {}

      specific_cuda() : block_stream_dth(0), block_stream_htd(0) ,
                      buffers() , blocksize(0) , allocated(false)
      {}

      ~specific_cuda()
      {
        allocated = false;
      }

      void swap(specific_cuda &)
      {

      }

      inline void reset()
      {
        allocated = false ;

        for(std::size_t i = 0 ; i < block_stream_dth.size() ; ++i )
        {
          block_stream_dth[i] = false;
          block_stream_htd[i] = false;
        }
      }

      template<typename In>
      inline void allocate(In const& in , std::size_t blocksize_ , std::size_t nstreams, std::size_t s)
      {
        if (!allocated)
        {
          blocksize = blocksize_ ;
          std::size_t num = s / blocksize_  ;
          block_stream_dth.resize(num);
          block_stream_htd.resize(num);
          auto loc_ = locality(in);
          buffers.allocate(blocksize, nstreams , loc_);
          allocated = true;
        }
      }

      template<class In, class Stream>
      inline void transfer_htd( In & in, int blockid, Stream & stream ,std::size_t streamid
                              , std::size_t leftover = 0 )
      {
        auto loc_ = locality(in);
        transfer_htd(in,blockid,stream,streamid,leftover, loc_);
      }

      template<class In, class Stream>
      inline void transfer_htd( In & in, int blockid, Stream & stream ,std::size_t streamid
                              , std::size_t leftover , nt2::host_ &)
      {
        std::size_t sizeb = blocksize;
        if(leftover !=0) sizeb = leftover ;

        if( block_stream_htd[blockid] == false )
        {
        block_stream_htd[blockid] = true;
        buffers.copy_hostpinned(in, streamid, sizeb , blockid);

        CUDA_ERROR(cudaMemcpyAsync( buffers.get_device(streamid)
                                  , buffers.get_host(streamid)
                                  , sizeb* sizeof(T)
                                  , cudaMemcpyHostToDevice
                                  , stream
                  ));
        cudaStreamSynchronize(stream);
        }

      }

      template<class In, class Stream>
      inline void transfer_htd( In & in, int blockid, Stream & stream ,std::size_t streamid
                              , std::size_t leftover , nt2::pinned_ & )
      {
        std::size_t sizeb = blocksize;
        if(leftover !=0) sizeb = leftover ;

        if( block_stream_htd[blockid] == false )
        {
        block_stream_htd[blockid] = true;

        CUDA_ERROR(cudaMemcpyAsync( buffers.get_device(streamid)
                                  , in.data()
                                  , sizeb* sizeof(T)
                                  , cudaMemcpyHostToDevice
                                  , stream
                  ));
        cudaStreamSynchronize(stream);
        }

      }

      template<class Out, class Stream>
      inline void transfer_dth( Out & out , int blockid, Stream & stream  ,std::size_t streamid
                              , std::size_t leftover = 0)
      {
        auto loc_ = locality(out);
        transfer_dth(out,blockid,stream,streamid,leftover,loc_);
      }

      template<class Out, class Stream>
      inline void transfer_dth( Out & out , int blockid, Stream & stream  ,std::size_t streamid
                              , std::size_t leftover , nt2::host_ &)
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
          cudaStreamSynchronize(stream);
          buffers.copy_host(out, streamid, sizeb , blockid);
        }
      }


      template<class Out, class Stream>
      inline void transfer_dth( Out & out , int blockid, Stream & stream  ,std::size_t streamid
                              , std::size_t leftover , nt2::pinned_ &)
      {
        std::size_t sizeb = blocksize;
        if(leftover !=0) sizeb = leftover ;

        if(block_stream_dth[blockid] == false )
        {
          CUDA_ERROR(cudaMemcpyAsync( out.data()
                          , buffers.get_device(streamid)
                          , sizeb * sizeof(T)
                          , cudaMemcpyDeviceToHost
                          , stream
                    ));

          block_stream_dth[blockid] = true;
          cudaStreamSynchronize(stream);

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
