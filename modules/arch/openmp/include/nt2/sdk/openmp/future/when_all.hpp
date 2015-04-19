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
    template <typename T>
    details::openmp_future< std::vector< details::openmp_shared_future<T> > >
    call( std::vector< details::openmp_future<T> > & lazy_values )
    {
      typedef typename std::vector<
         details::openmp_shared_future<T>
      > whenall_vector;

      typedef typename details::openmp_future< whenall_vector >
      whenall_future;

      whenall_vector result ( lazy_values.size() );

      for(std::size_t i=0; i<lazy_values.size(); i++)
        result[i] = std::move(lazy_values[i]);

      details::openmp_task_wrapper<
        std::function< whenall_vector(whenall_vector) >
      , whenall_vector
      , whenall_vector
      >
      packaged_task(
        []( whenall_vector result_ ){
             return result_;
          }
        , std::move(result)
        );

      );

      whenall_future future_res(packaged_task.get_future());

      bool * next( future_res.ready_.get() );
      bool * deps[size];

      for (std::size_t i=0; i<size; i++)
      {
        deps[i] = lazy_values[i].ready_.get();
      }

      static_cast<void>(deps);

      #pragma omp task \
      firstprivate(packaged_task, next, deps) \
      depend( in : deps[0:size] ) \
      depend( out : next )
      {
        packaged_task();
        *next = true;
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

#define NT2_FUTURE_FORWARD_ARGS0(z,n,t) details::openmp_shared_future<A##n>
#define NT2_FUTURE_FORWARD_ARGS1(z,n,t) details::openmp_future<A##n> & a##n
#define NT2_FUTURE_FORWARD_ARGS2(z,n,t) std::move(a##n)
#define NT2_FUTURE_FORWARD_ARGS3(z,n,t) bool * r##n = POINT(a##n,ready_).get();
#define NT2_FUTURE_FORWARD_ARGS4(z,n,t) boost::ignore_unused(r##n);

    template< BOOST_PP_ENUM_PARAMS(N, typename A) >
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

      details::openmp_task_wrapper<
        std::function<whenall_tuple(whenall_tuple)>
      , whenall_tuple
      , whenall_tuple
      >
      packaged_task( [](whenall_tuple result_)
                     { return result_;
                     }
                   , std::move(result)
                   );

      whenall_future future_res (packaged_task.get_future());

      bool * next( future_res.ready_.get() );

      BOOST_PP_REPEAT(N, NT2_FUTURE_FORWARD_ARGS3, ~)
      BOOST_PP_REPEAT(N, NT2_FUTURE_FORWARD_ARGS4, ~)
      #pragma omp task \
      firstprivate(packaged_task, next, BOOST_PP_ENUM_PARAMS(N,r) ) \
      depend( in : BOOST_PP_ENUM_PARAMS(N,r) ) \
      depend( out : next )
      {
        packaged_task();
        *next = true;
      }

      return future_res;
    }

#undef NT2_FUTURE_FORWARD_ARGS0
#undef NT2_FUTURE_FORWARD_ARGS1
#undef NT2_FUTURE_FORWARD_ARGS2
#undef NT2_FUTURE_FORWARD_ARGS3
#undef NT2_FUTURE_FORWARD_ARGS4
#undef POINT
#undef N

#endif
