//==============================================================================
//         Copyright 20014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_META_DEVICE_HPP_INCLUDED
#define NT2_SDK_META_DEVICE_HPP_INCLUDED

#if defined(NT2_HAS_CUDA)

#include <type_traits>
#include <nt2/core/settings/settings.hpp>
#include <nt2/core/settings/add_settings.hpp>
#include <nt2/include/functions/copy.hpp>
#include <nt2/sdk/meta/cuda_alloc.hpp>

namespace nt2 { namespace meta
  {

  template<class In_,class Enableif = void>
  struct impl
  {
    using value_type = typename In_::value_type;
    using settings   = typename add_settings<typename In_::settings_type,nt2::device_>::type;
    using type = nt2::container::table<value_type , settings>;

    static type init1(In_ & in )
    {
      type result(in.extent());
      nt2::memory::copy(in, result, locality(in),locality(result));

      return result;
    }
  };

  template<class In_>
  struct impl<In_ ,typename std::enable_if< (cuda_alloc_type == cudaHostAllocMapped)
                                          && std::is_same<typename In_::allocator_type,nt2::memory::cuda_pinned_<typename In_::value_type> >::value
                                          >::type
            >
  {
    using value_type = typename In_::value_type;
    using settings   = typename add_settings<typename In_::settings_type,nt2::device_>::type ;
    using table_type = nt2::container::table<value_type , settings>;
    using type = nt2::container::view<table_type>;

    static type init1(In_ & in)
    {
      type result;
      // value_type * out = nullptr;
      // cudaHostGetDevicePointer( (void **) &out, (void*) in.data() ,0 );
      result.reset(in.data(),in.extent() );

    return result;
    }
  };

  template<class In, class Enable = void>
  struct as_device
  {
    using type = typename impl<In>::type;

    static type init(In & in)
    {
      return impl<In>::init1(in);
    }

  };

  template<class In>
  struct as_device<In, typename std::enable_if< is_on_device<In>::value>::type  >
  {
    using type = In&;

    static type init(In & in)
    {
      return in;
    }
  };

  template<class In, class Loc, class Enable = void>
  struct as_host
  {
    using value_type = typename In::value_type;
    using settings = typename add_settings<typename In::settings_type,nt2::host_>::type ;
    using settings1 = typename add_settings<settings,Loc>::type ;
    using type = nt2::container::table<value_type , settings1> ;

    static type init(In & in)
    {
      type out = in;
      return out;
    }

  };

  template<class In, class Loc>
  struct as_host<In, Loc, typename std::enable_if< is_on_host<In>::value >::type  >
  {
    using type = In&;

    static type init(In & in)
    {
      return in;
    }
  };

  template<class In, class Out, class Enable = void>
  struct as_host_inout
  {
    static void init(In & in, Out & out )
    {
      out = in;
    }

  };

  template<class In, class Out>
  struct as_host_inout<In,Out
                      ,typename std::enable_if< (cuda_alloc_type == cudaHostAllocMapped)
                                              && std::is_same<typename Out::allocator_type
                                                              ,nt2::memory::cuda_pinned_<typename Out::value_type>
                                                              >::value
                                             >::type
                      >
  {
    static void init(In & in, Out & out )
    {
      cudaDeviceSynchronize();
    }

  };


  template<class In, class Out, class Enable = void>
  struct as_container
  {
    static void init(In & in, Out & out)
    {
      nt2::memory::copy(in, out , nt2::device_() , nt2::host_() );
    }
  };

  template<class In, class Out >
  struct as_container<In, Out , typename std::enable_if<  std::is_same<In,Out>::value>::type >
  {
    static void init(In & , Out& ) {}
  };

}
}

namespace nt2
{
  template<class A, class B>
  void device_swap(A& a, B & b)
  {
    meta::as_container<typename meta::as_device<B>::type,B>::init(a,b);
  }

  /*!
    return a Sequence on the device
  **/
  template<class In>
  auto to_device(In & a) -> decltype(meta::as_device<In>::init(a))
  {
    return meta::as_device<In>::init(a);
  }

  /*!
    return a Sequence on the host
  **/
  template<class LOC = void, class In>
  auto to_host(In & a ) -> decltype(meta::as_host<In,LOC>::init(a))
  {
    return meta::as_host<In,LOC>::init(a);
  }

  template<class In, class Out>
  void to_host(In & a, Out & b )
  {
    meta::as_host_inout<In,Out>::init(a,b);
  }


}

#endif
#endif

