//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_FUTURE_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_FUTURE_HPP_INCLUDED

#include <type_traits>
#include <utility>
#include <vector>
#include <future>
#include <tuple>

namespace nt2
{
    template<class Arch, class result_type>
    struct make_future;

    template<class Arch, class result_type>
    struct make_shared_future;

    template<class Arch>
    struct async_impl;

    template<class Arch, typename result_type>
    struct make_ready_future_impl;

    template< typename Arch, typename result_type>
    inline auto make_ready_future(result_type value)
    -> decltype( make_ready_future_impl<Arch,result_type>()
                .call(std::move(value))
               )
    {
       return make_ready_future_impl<Arch,result_type>()
              .call(std::move(value) );
    }

    template<class Arch>
    struct when_all_impl;

    template< typename Arch,typename ... A>
    inline auto when_all(A && ... a)
    -> decltype( when_all_impl<Arch>().call( std::forward<A>(a)... ) )
    {
      return when_all_impl<Arch>().call( std::forward<A>(a)... );
    }

    template< typename Arch,typename F, typename ... A >
    inline auto async( F && f, A && ... a)
    -> decltype( async_impl<Arch>().call( std::forward<F>(f)
                                        , std::forward<A>(a)...
                                        )
               )
    {
        return async_impl<Arch>().call( std::forward<F>(f)
                                      , std::forward<A>(a)...
                                      );
    }
}


#endif
