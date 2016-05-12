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
#include <nt2/core/settings/add_settings.hpp>
#include <nt2/include/functions/copy.hpp>
#include <nt2/sdk/meta/cuda_alloc.hpp>

namespace nt2 { namespace meta
  {

  template<class In_,class Enableif = void>
  struct impl
  {
    using value_type = typename In_::value_type;
    using settings   = typename add_settings<nt2::device_,typename In_::nt2_expression>::type;
    using table_type = nt2::container::table<value_type , settings>;

    static table_type init1(In_ & in )
    {
      table_type result(in.extent());
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
    using settings   = typename add_settings<nt2::device_,typename In_::nt2_expression>::type ;
    using table_type = nt2::container::table<value_type , settings>;
    using type = nt2::container::view<table_type>;

    static type init1(In_ & in)
    {
      type result;
      value_type * out = nullptr;
      cudaHostGetDevicePointer( (void **) &out, (void*) in.data() ,0 );
      result.reset(out,in.extent() );

    return result;
    }
  };

  template<class In, class Enable = void>
  struct as_device
  {
    static auto init(In & in) -> decltype(impl<In>::init1(in))
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
    using settings = typename add_settings<Loc,typename In::nt2_expression>::type ;
    using table_type = nt2::container::table<value_type , settings> ;

    static table_type init(In & in)
    {
      table_type out = in;
      return out;
    }

  };

  template<class In, class Loc>
  struct as_host<In, Loc, typename std::enable_if< is_on_host<In>::value>::type  >
  {
    using type = In&;

    static type init(In & in)
    {
      return in;
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
  struct as_container<In, Out
                    , typename std::enable_if<std::is_same<In,Out>::value>::type
                     >
  {
    static void init(In& , Out& ) {}
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
  template<class A>
  auto to_device(A & a) -> decltype(meta::as_device<A>::init(a))
  {
    return meta::as_device<A>::init(a);
  }

  template<class B = nt2::host_ , class A>
  auto to_host(A & a ) -> decltype(meta::as_host<A,B>::init(a))
  {
    return meta::as_host<A,B>::init(a);
  }

}

#endif
#endif

