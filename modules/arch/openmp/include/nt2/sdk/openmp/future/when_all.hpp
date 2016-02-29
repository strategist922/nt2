//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#if !BOOST_PP_IS_ITERATING
#ifndef NT2_SDK_OPENMP_FUTURE_WHEN_ALL_HPP_INCLUDED
#define NT2_SDK_OPENMP_FUTURE_WHEN_ALL_HPP_INCLUDED

#if defined(_OPENMP) && _OPENMP >= 201307 /* OpenMP 4.0 */

#include <omp.h>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>

#include <nt2/sdk/shared_memory/future.hpp>
#include <nt2/sdk/openmp/future/details/openmp_future.hpp>
#include <nt2/sdk/openmp/future/details/openmp_shared_future.hpp>
#include <nt2/sdk/openmp/future/details/openmp_task_wrapper.hpp>

#include <vector>

namespace nt2
{
  template<class Site>
  struct when_all_impl< tag::openmp_<Site> >
  {
    template <typename T, template<typename> class Future >
    details::openmp_future< std::vector< details::openmp_shared_future<T> > >
    call( std::vector< Future<T> > & lazy_values )
    {
      typedef typename std::vector<
         details::openmp_shared_future<T>
      > whenall_vector;

      typedef typename details::openmp_future< whenall_vector >
      whenall_future;

      std::size_t size (lazy_values.size() );

      whenall_vector result (size);

      for(std::size_t i=0; i<size; i++)
        result[i] = lazy_values[i];

      auto task = []( whenall_vector result_ ){return result_;};

      details::openmp_task_wrapper<
        decltype(task)
      , whenall_vector
      , whenall_vector
      >
      packaged_task(std::move(task), std::move(result));

      whenall_future future_res(packaged_task.get_future());

      int * next( future_res.ready_.get() );
      int * deps[size];

      for (std::size_t i=0; i<size; i++)
      {
        deps[i] = lazy_values[i].ready_.get();
      }

      static_cast<void>(deps);

      #pragma omp task \
      firstprivate(size, packaged_task, next, deps) \
      depend( in : deps[0:size][0:1]) \
      depend( out : next[0:1] )
      {
        packaged_task();
        *next = 1;
      }

      return future_res;
    }

#define BOOST_PP_ITERATION_PARAMS_1 (3, \
    ( 1, BOOST_DISPATCH_MAX_ARITY, \
      "nt2/sdk/openmp/future/when_all.hpp") \
    )

#include BOOST_PP_ITERATE()
};
}

#endif
#endif

#else

#define N BOOST_PP_ITERATION()

#define POINT(a,b) a.b
#define BRACKET(a,b) a[b]

#define NT2_FUTURE_FORWARD_ARGS0(z,n,t) details::openmp_shared_future<A##n>
#define NT2_FUTURE_FORWARD_ARGS1(z,n,t) Future<A##n> & a##n
#define NT2_FUTURE_FORWARD_ARGS2(z,n,t) details::openmp_shared_future<A##n>(a##n)
#define NT2_FUTURE_FORWARD_ARGS3(z,n,t) int * r##n = POINT(a##n,ready_).get();
#define NT2_FUTURE_FORWARD_ARGS4(z,n,t) boost::ignore_unused(r##n);
#define NT2_FUTURE_FORWARD_ARGS5(z,n,t) BRACKET(r##n,0:1)

    template< BOOST_PP_ENUM_PARAMS(N, typename A)
            , template<typename> class Future
            >
    typename details::openmp_future<
    std::tuple< BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS0, ~) >
    >
    call( BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS1, ~) )
    {
      typedef typename std::tuple< BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS0, ~) >
      whenall_tuple;

      typedef typename details::openmp_future< whenall_tuple >
      whenall_future;

      whenall_tuple result
        = std::make_tuple<
                          BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS0, ~)
                          >
                       ( BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS2, ~) );

      auto task = [](whenall_tuple result_) { return result_;};

      details::openmp_task_wrapper<
        decltype(task)
      , whenall_tuple
      , whenall_tuple
      >
      packaged_task( std::move(task), std::move(result) );

      whenall_future future_res (packaged_task.get_future());

      int * next( future_res.ready_.get() );

      BOOST_PP_REPEAT(N, NT2_FUTURE_FORWARD_ARGS3, ~)
      BOOST_PP_REPEAT(N, NT2_FUTURE_FORWARD_ARGS4, ~)
      #pragma omp task \
      firstprivate(packaged_task, next, BOOST_PP_ENUM_PARAMS(N,r) ) \
      depend( in : BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS5, ~) ) \
      depend( out : next[0:1] )
      {
        packaged_task();
        *next = 1;
      }

      return future_res;
    }

#undef NT2_FUTURE_FORWARD_ARGS0
#undef NT2_FUTURE_FORWARD_ARGS1
#undef NT2_FUTURE_FORWARD_ARGS2
#undef NT2_FUTURE_FORWARD_ARGS3
#undef NT2_FUTURE_FORWARD_ARGS4
#undef POINT
#undef BRACKET
#undef N

#endif
