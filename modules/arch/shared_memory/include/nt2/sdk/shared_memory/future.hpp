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
#include <nt2/sdk/shared_memory/details/nt2_future.hpp>
#include <nt2/sdk/shared_memory/details/nt2_shared_future.hpp>
#include <nt2/sdk/shared_memory/details/wait_tuple_of_futures.hpp>

namespace nt2
{

    // Default definition of future if no backend found
    template<class Arch, class result_type>
    struct make_future
    {
        typedef typename details::nt2_future<result_type> type;
    };

    // Default definition of shared_future if no backend found
    template<class Arch, class result_type>
    struct make_shared_future
    {
        typedef typename std::shared_future<result_type> type;
    };

    // Default implementation of async if no backend found
    template<class Arch>
    struct async_impl
    {
        template< typename F, typename ... A >
        inline details::nt2_future<
                 typename std::result_of< F(A...)>::type
               >
        call(F && f, A && ... a)
        {
            return std::async( std::forward<F>(f)
                             , std::forward<A>(a) ...
                             );
        }
    };

    // Default implementation of make_ready_future if no backend found
    template<class Arch, typename result_type>
    struct make_ready_future_impl
    {
        inline details::nt2_future<result_type> call(result_type && value)
        {
          std::promise<result_type> promise;
          details::nt2_future<result_type> future_res ( promise.get_future() );
          promise.set_value(value);
          return future_res;
        }
    };

    // Default implementation of when_all if no backend found
    template<class Arch>
    struct when_all_impl
    {
        template< typename ... A >
        inline details::nt2_future<
            std::tuple< details::nt2_shared_future<A> ... >
            >
        call(details::nt2_future<A> & ...a )
        {
          typedef std::tuple< details::nt2_shared_future<A> ... >
          when_all_tuple;

          return  std::async(
            [&](){  when_all_tuple res = std::make_tuple<
                                         details::nt2_shared_future<A> ...
                                         >( a.share() ... );

                    details::wait_tuple_of_futures< sizeof...(A) >()
                    .call(res);

                    return res;
                 }
            );
        }

        template <typename T>
        details::nt2_future< std::vector< details::nt2_shared_future<T> > >
        call( std::vector< details::nt2_future<T> > & lazy_values )
        {
          typedef typename std::vector< details::nt2_shared_future<T> >
          whenall_vector;

          whenall_vector returned_lazy_values ( lazy_values.size() );

          for(std::size_t i=0; i< lazy_values.size(); i++)
          {
            returned_lazy_values[i] = lazy_values[i].share();
          }

          return  std::async(
            [& returned_lazy_values]() -> whenall_vector
            {
              for (std::size_t i=0; i<returned_lazy_values.size(); i++)
              {
                  returned_lazy_values[i].wait();
              }
              return returned_lazy_values;
            }
          );
        }
    };

    template< typename Arch, typename result_type>
    inline auto make_ready_future(result_type value)
    -> decltype( make_ready_future_impl<Arch,result_type>()
                .call(std::move(value))
               )
    {
       return make_ready_future_impl<Arch,result_type>()
              .call(std::move(value) );
    }



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
