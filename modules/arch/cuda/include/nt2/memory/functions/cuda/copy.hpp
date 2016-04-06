//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_MEMORY_FUNCTIONS_CUDA_COPY_HPP_INCLUDED
#define NT2_MEMORY_FUNCTIONS_CUDA_COPY_HPP_INCLUDED

#include <boost/dispatch/details/auto_decltype.hpp>

#if defined(NT2_HAS_CUDA)

#include <nt2/sdk/memory/cuda/buffer.hpp>
#include <nt2/core/settings/locality.hpp>
#include <cuda_runtime.h>

namespace nt2 { namespace memory
{
  template<typename In, typename Out>
  struct copy_;

  template<> struct copy_<device_,host_>
  {
    static BOOST_FORCEINLINE
    BOOST_AUTO_DECLTYPE mode()
    BOOST_AUTO_DECLTYPE_BODY ( cudaMemcpyDeviceToHost )
  };

  template<> struct copy_<host_,device_>
  {
    static BOOST_FORCEINLINE
    BOOST_AUTO_DECLTYPE mode()
    BOOST_AUTO_DECLTYPE_BODY ( cudaMemcpyHostToDevice )
  };

  template<> struct copy_<host_,host_>
  {
    static BOOST_FORCEINLINE
    BOOST_AUTO_DECLTYPE mode()
    BOOST_AUTO_DECLTYPE_BODY ( cudaMemcpyHostToHost )
  };

  template<> struct copy_<device_,device_>
  {
    static BOOST_FORCEINLINE
    BOOST_AUTO_DECLTYPE mode()
    BOOST_AUTO_DECLTYPE_BODY ( cudaMemcpyDeviceToDevice )
  };

  template<typename In, typename Out, typename HDI, typename HDO>
  inline void copy( In const& a, Out& b , HDI const& , HDO const&
                  , cudaStream_t stream = 0)
  {
    using T = typename Out::value_type;

//TODO
    CUDA_ERROR(cudaMemcpyAsync( (T*)b.data()
                              , a.data()
                              , a.size()* sizeof(T)
                              , copy_<HDI,HDO>::mode()
                              , stream
                              ));
  }

  template<typename T, class C1>
  inline void copy(cuda_buffer<T> const& a, C1 &b, cudaStream_t stream = 0)
  {
    copy(a,b,nt2::device_{},nt2::host_{},stream);
  }

  template< class C1,typename T>
  inline void copy(C1 const& a, cuda_buffer<T> &b, cudaStream_t stream = 0)
  {
    b.resize(a.size());
    copy(a,b,nt2::host_{},nt2::device_{},stream);
  }

  template<typename T>
  inline void copy( cuda_buffer<T> const& a, cuda_buffer<T> & b
                  , cudaStream_t stream = 0 )
  {
    b.resize(a.size());
    copy(a,b,nt2::device_{},nt2::device_{},stream);
  }


}}

namespace nt2 { namespace container
{

  template<class A0, class A1>
  typename std::enable_if< meta::is_on_device<A0>::value
                        || meta::is_on_device<A1>::value>::type
  assign_swap(A0& a0, A1& a1)
  {
    device_swap(a0,a1);
  }

}}


#endif

#endif
