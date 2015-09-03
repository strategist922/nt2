//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_OPENCL_SETTINGS_SPECIFIC_OPENCL_HPP_INCLUDED
#define NT2_SDK_OPENCL_SETTINGS_SPECIFIC_OPENCL_HPP_INCLUDED

#ifdef NT2_HAS_OPENCL

#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <cstring>
#include <iostream>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>

namespace nt2{ namespace details
  {
  namespace compute = boost::compute;
    template<typename T>
    struct cl_buffers
    {
      std::vector<compute::vector<T, compute::pinned_allocator<T> > > host_pinned;
      std::vector<compute::vector<T> > device;
      std::size_t size;

      cl_buffers() : host_pinned(0), device(0) ,size(0) {}
      cl_buffers(std::size_t size_,std::size_t nstreams)
      {
        size = size_ ;
        allocate(size,nstreams);
      }

      void allocate(std::size_t size_,std::size_t nstreams, compute::command_queue *queues)
      {
        size = size_;
        if(size != device.size())
        {
//          host_pinned.resize(nstreams);
//          device.resize(nstreams);
          host_pinned.clear();
          device.clear();
          for ( std::size_t i = 0 ; i < nstreams ; ++i ) {
//            host_pinned[i].resize(size_, queues[i]);
//            device[i].resize(size_, queues[i]);
            device.push_back(compute::vector<T>(size_, queues[i].get_context()));
//            host_pinned.push_back( compute::vector<T, compute::pinned_allocator<T> >(size_, queues[i].get_context()) );
//compute::vector<T, compute::pinned_allocator<T> > test(size_, queues[i].get_context());
//host_pinned.push_back(test);
          }
        }
      }

      compute::vector<T> & get_host(std::size_t indx)
      {
        return host_pinned[indx];
      }

      compute::vector<T> & get_device(std::size_t indx)
      {
        return device[indx];
      }

      template<class Container>
      void copy_dev2host(Container & c, std::size_t streamid, std::size_t sizeb
                        , int blockid
                        , compute::command_queue queue)
      {
        queue.enqueue_read_buffer(
                                  device[streamid].get_buffer()
                                 , blockid * sizeb
                                 , sizeb// * sizeof(T)
                                 , c.data()
                                 );
      }

      //TODO: Confirm that you want to use sizeb and not sizeb*sizeof(T)
      template<class Container>
      void copy_host2dev(Container & c, std::size_t streamid, std::size_t sizeb
                        , int blockid
                        , compute::command_queue queue)
      {
//        device[streamid].resize(sizeb);
        queue.enqueue_write_buffer(
                                  device[streamid].get_buffer()
                                 , blockid * sizeb
                                 , sizeb// * sizeof(T)
                                 , c.data()
                                 );
      }

//      void copy_host2dev(compute::vector<T> & c, std::size_t streamid, std::size_t sizeb
//                        , int blockid
//                        , compute::command_queue queue)
//      {
//        device[streamid].resize(sizeb);
//        queue.enqueue_write_buffer(
//                                  device[streamid].get_buffer()
//                                 , blockid * sizeb
//                                 , sizeb// * sizeof(T)
//                                 , c.data().get_buffer()
//                                 );
//      }
//
      template<class Container>
      void copy_hostpinned(Container & c,std::size_t streamid, std::size_t sizeb
                          , int blockid
                          , compute::command_queue queue)
      {
        host_pinned[streamid].resize(sizeb, queue);
        compute::copy( c.data() + blockid * size
                     , c.data() + blockid * size + sizeb*sizeof(T)
                     , host_pinned[streamid].begin()
                     );
      }


      template<class Container>
      void copy_host(Container & c, std::size_t streamid, std::size_t sizeb
                    , int blockid
                    , compute::command_queue queue)
      {
        compute::copy( host_pinned[streamid].begin()
                     , host_pinned[streamid].begin() + sizeb*sizeof(T)
                     , c.data() + blockid * size
                     );
      }

      ~cl_buffers()
      {
      }

    };// end class cl_buffers

    template<typename Arch, typename T>
    struct specific_opencl
    {
      using btype = boost::dynamic_bitset<>;
      std::size_t blocksize;
      bool allocated;
      btype block_stream_dth;
      btype block_stream_htd;
      cl_buffers<T> buffers;

      inline void synchronize() {}

      specific_opencl() : block_stream_dth(1),
                      block_stream_htd(1) ,
                      buffers{} ,
                      blocksize(0) ,
                      allocated(false)
      {
      }

      ~specific_opencl()
      {
        block_stream_dth.reset();
        block_stream_htd.reset();
        allocated = false;
      }

      void swap(specific_opencl &)
      {

      }

      inline void allocate(std::size_t blocksize_ , std::size_t nstreams, std::size_t s, compute::command_queue *queues)
      {
        if (!allocated)
        {
          blocksize = blocksize_ ;
          std::size_t num = s / blocksize_  ;
          block_stream_dth.resize(num);
          block_stream_htd.resize(num);
          buffers.allocate(s, nstreams, queues);
          allocated = true;
        }
      }

      template<class In, class Stream/*, class Set*/>
      inline void transfer_htd( In & in, int blockid, Stream & stream ,std::size_t streamid
//                              , Set & addr
                              , std::size_t leftover = 0)
      {
        std::size_t sizeb = blocksize;
        if(leftover !=0) sizeb = leftover ;

        if( block_stream_htd[blockid] == false
          ) {
          buffers.copy_host2dev(in, streamid, sizeb, blockid, stream);
          block_stream_htd[blockid] = true;
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
          buffers.copy_dev2host(out, streamid, sizeb, blockid, stream);
          block_stream_dth[blockid] = true;
        }
      }

      inline compute::vector<T> & data(std::size_t i)
      {
        return buffers.get_device(i);
      }

      template<typename Container>
      inline void copy_host(Container & c, std::size_t streamid,  std::size_t sizeb
                           ,  std::size_t blockid, compute::command_queue queue )
      {
        buffers.copy_host(c, streamid, sizeb , blockid, queue);;
      }

    };

  }
}

#endif

#endif
