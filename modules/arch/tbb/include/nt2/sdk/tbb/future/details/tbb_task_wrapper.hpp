//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_TBB_FUTURE_DETAILS_TBB_TASK_WRAPPER_HPP_INCLUDED
#define NT2_SDK_TBB_FUTURE_DETAILS_TBB_TASK_WRAPPER_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <tbb/tbb.h>
#include <tbb/flow_graph.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <tuple>

namespace nt2
{
  namespace details
  {

    template< typename T>
    struct tbb_future;

    template<int ... Indices>
    struct tbb_sequence{
      template<class F, typename ... A>
      typename std::result_of<F(A...)>::type
      apply(F& f, std::tuple<A...> & tuple)
      {
        return f( std::get<Indices>(tuple)... );
      }
    };

    template<int N, int ...S>
    struct generate_tbb_sequence
    : generate_tbb_sequence<N-1, N-1, S...> { };

    template<int ...S>
    struct generate_tbb_sequence<0, S...> {
      typedef tbb_sequence<S...> type;
    };

    template< class F, typename result_type, typename ... A>
    struct tbb_task_wrapper
    {
      typedef typename generate_tbb_sequence<sizeof...(A)>::type seq;
      typedef typename std::promise<result_type> promise_type;

      tbb_task_wrapper( F && f , A&& ... a)
      : f_(std::forward<F>(f))
      , a_( std::make_tuple(std::forward<A>(a) ...) )
      {}

      tbb_task_wrapper( tbb_task_wrapper const & other )
      : f_(other.f_)
      , promise_(std::move(other.promise_))
      , a_(other.a_)
      {}

      std::future<result_type> get_future()
      {
        return promise_.get_future();
      }

      void operator()(const tbb::flow::continue_msg )
      {
        promise_.set_value( seq().apply(f_,a_) );
      }

      F f_;
      mutable promise_type promise_;
      std::tuple < A ... > a_;
    };

    // Specialization for continuation wrapper

    template< class F, typename result_type, typename T>
    struct tbb_task_wrapper<F, result_type, details::tbb_future<T> >
    {
      typedef typename std::promise<result_type> promise_type;

      tbb_task_wrapper( F && f
                      , details::tbb_future<T> && a
                      )
      : f_(std::forward<F>(f))
      , a_( std::forward< details::tbb_future<T> >(a))
      {}

      tbb_task_wrapper( tbb_task_wrapper const & other )
      : f_(other.f_)
      , promise_( std::move(other.promise_) )
      , a_( std::move(other.a_) )
      {}

      std::future<result_type> get_future()
      {
        return promise_.get_future();
      }

      void operator()(const tbb::flow::continue_msg )
      {
        promise_.set_value( f_(std::move(a_)) );
      }

      F f_;
      mutable promise_type promise_;
      mutable details::tbb_future<T> a_;
    };
  }
}

#endif
#endif
