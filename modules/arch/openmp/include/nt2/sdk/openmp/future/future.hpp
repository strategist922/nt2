//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_OPENMP_FUTURE_FUTURE_HPP_INCLUDED
#define NT2_SDK_OPENMP_FUTURE_FUTURE_HPP_INCLUDED

#if defined(_OPENMP) && _OPENMP >= 201307 /* OpenMP 4.0 */

#include <omp.h>
#include <unistd.h>

#include <nt2/sdk/shared_memory/future.hpp>
#include <nt2/sdk/openmp/future/details/openmp_future.hpp>
#include <nt2/sdk/openmp/future/details/openmp_shared_future.hpp>
#include <nt2/sdk/openmp/future/details/openmp_task_wrapper.hpp>

#include <type_traits>

namespace nt2
{
  namespace tag
  {
    template<class T> struct openmp_;
  }

  template<class Site, class result_type>
  struct make_future<tag::openmp_<Site> , result_type>
  {
    typedef typename nt2::details::openmp_future<result_type> type;
  };

  template<class Site, class result_type>
  struct make_shared_future<tag::openmp_<Site> , result_type>
  {
    typedef typename nt2::details::openmp_shared_future<result_type> type;
  };


  template< class Site, class result_type>
  struct make_ready_future_impl< tag::openmp_<Site>, result_type>
  {
    inline details::openmp_future<result_type>
    call(result_type && value)
    {
      std::promise<result_type> promise;
      details::openmp_future<result_type> future_res ( promise.get_future() );
      promise.set_value(value);

      int * next( future_res.ready_.get() );

      #pragma omp task  \
      firstprivate(next)\
      depend(out: next[0:1])
      {
        *next = 1;
      }

      return future_res;
    }
  };

  template<class Site>
  struct async_impl< tag::openmp_<Site> >
  {
    template< typename F , typename ... A>
    inline details::openmp_future<
              typename std::result_of< F(A...)>::type
           >
    call(F&& f, A&& ... a)
    {
      typedef typename std::result_of< F(A...)>::type result_type;
      typedef typename details::openmp_future<result_type> async_future;

      details::openmp_task_wrapper< F, result_type, A ... >
      packaged_task
      ( std::forward<F>(f)
      , std::forward<A>(a)...
      );

      async_future future_res( packaged_task.get_future() );

      int * next( future_res.ready_.get() );

      #pragma omp task \
      firstprivate(packaged_task,next) \
      depend(out: next[0:1])
      {
        packaged_task();
        *next = 1;
      }

      return future_res;
    }

  };
}


#endif
#endif
