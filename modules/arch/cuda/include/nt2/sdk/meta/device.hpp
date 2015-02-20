//==============================================================================
//         Copyright 20014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_META_DEVICE_HPP_INCLUDED
#define NT2_SDK_META_DEVICE_HPP_INCLUDED

#include <type_traits>
#include <nt2/core/settings/add_settings.hpp>
#include <nt2/include/functions/copy.hpp>

#if defined(NT2_HAS_CUDA)

namespace nt2 { namespace meta
  {

 template<class In, class Enable = void>
  struct as_device
  {
    using type = nt2::memory::cuda_buffer<typename In::value_type> ;

    static type init(In & in)
    {
      type result(in.size());
      nt2::memory::copy(in, result);
      return result;
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
}

#endif
#endif

