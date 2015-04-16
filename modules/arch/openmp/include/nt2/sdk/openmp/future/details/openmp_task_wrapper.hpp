//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_OPENMP_FUTURE_DETAILS_OPENMP_TASK_WRAPPER_HPP_INCLUDED
#define NT2_SDK_OPENMP_FUTURE_DETAILS_OPENMP_TASK_WRAPPER_HPP_INCLUDED

#if defined(_OPENMP) && _OPENMP >= 201307 /* OpenMP 4.0 */

#include <omp.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <tuple>

namespace nt2
{
  namespace details
  {

    template< typename T>
    struct openmp_future;

    template<int ... Indices>
    struct openmp_sequence{
      template<class F, typename ... A>
      typename std::result_of<F(A...)>::type
      apply(F& f, std::tuple<A...> & tuple)
      {
        return f( std::get<Indices>(tuple)... );
      }
    };

    template<int N, int ...S>
    struct generate_openmp_sequence
    : generate_openmp_sequence<N-1, N-1, S...> { };

    template<int ...S>
    struct generate_openmp_sequence<0, S...> {
      typedef openmp_sequence<S...> type;
    };

    template< class F, typename result_type, typename ... A>
    struct openmp_task_wrapper
    {
      typedef typename generate_openmp_sequence<sizeof...(A)>::type seq;
      typedef typename std::promise<result_type> promise_type;

      openmp_task_wrapper( F && f, A&& ... a)
      : f_(std::forward<F>(f))
      , a_( std::make_tuple(std::forward<A>(a) ...) )
      {}

      openmp_task_wrapper( openmp_task_wrapper const & other )
      : f_(other.f_)
      , promise_(std::move(other.promise_))
      , a_(other.a_)
      {}

      std::future<result_type> get_future()
      {
        return promise_.get_future();
      }

      void operator()()
      {
        promise_.set_value( seq().apply(f_,a_) );
      }

      F f_;
      mutable promise_type promise_;
      std::tuple < A ... > a_;
    };

    // Specialization for continuation wrapper

    template< class F, typename result_type, typename T>
    struct openmp_task_wrapper<F, result_type, details::openmp_future<T> >
    {
      typedef typename std::promise<result_type> promise_type;

      openmp_task_wrapper( F && f
                         , details::openmp_future<T> && a
                         )
      : f_(std::forward<F>(f))
      , a_( std::forward< details::openmp_future<T> >(a))
      {}

      openmp_task_wrapper( openmp_task_wrapper const & other )
      : f_(other.f_)
      , promise_( std::move(other.promise_) )
      , a_( std::move(other.a_) )
      {}

      std::future<result_type> get_future()
      {
        return promise_.get_future();
      }

      void operator()()
      {
        promise_.set_value( f_(std::move(a_)) );
      }

      F f_;
      mutable promise_type promise_;
      mutable details::openmp_future<T> a_;
    };
  }
}

#endif
#endif
