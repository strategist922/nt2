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
#ifndef NT2_SDK_TBB_FUTURE_WHEN_ALL_HPP_INCLUDED
#define NT2_SDK_TBB_FUTURE_WHEN_ALL_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <tbb/tbb.h>
#include <tbb/flow_graph.h>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>
#include <boost/dispatch/details/parameters.hpp>

#include <nt2/sdk/tbb/future/future.hpp>
#include <nt2/sdk/tbb/future/details/tbb_future.hpp>
#include <nt2/sdk/tbb/future/details/empty_body.hpp>

#include <boost/tuple/tuple.hpp>

namespace nt2
{

    namespace details
    {
        struct empty_functor
        {
            void operator()() const
            {}
        };
    }

    template<class Site>
    struct when_all_impl< tag::tbb_<Site> >
    {
       typedef typename tbb::flow::continue_node<\
       tbb::flow::continue_msg> node_type;

        template <typename Future>
        details::tbb_future< std::vector<Future> >
        call( BOOST_FWD_REF(std::vector<Future>) lazy_values )
        {
          typedef typename std::vector<Future> result_type;

          details::tbb_future<result_type> future_res;

          details::empty_body f;
          node_type * c = new node_type( *future_res.getWork(), f );
          future_res.getTaskQueue()->push_back(c);

          for (std::size_t i=0; i<lazy_values.size(); i++)
          {
            tbb::flow::make_edge(*(lazy_values[i].node_),*c);
            future_res.attach_task(c);
          }

          future_res.res_ = boost::make_shared(new result_type(lazy_values));

          return future_res;
        }

#define BOOST_PP_ITERATION_PARAMS_1 (3, \
( 1, BOOST_DISPATCH_MAX_ARITY, \
"nt2/sdk/tbb/future/when_all.hpp") \
)

#include BOOST_PP_ITERATE()
    };
}

#endif
#endif

#else

#define N BOOST_PP_ITERATION()

#define POINT(a,b) a.b

#define NT2_FUTURE_TEMPLATE(z, n, t) details::tbb_future<A##n>
#define NT2_FUTURE_FORWARD_ARGS(z,n,t) details::tbb_future<A##n> const & a##n
#define NT2_FUTURE_FORWARD_ARGS1(z,n,t) tbb::flow::make_edge(*(POINT(a##n,node_)),*c);

        template< BOOST_PP_ENUM_PARAMS(N, typename A) >
        details::tbb_future<\
          boost::tuple<\
            BOOST_PP_ENUM(N,NT2_FUTURE_TEMPLATE, ~)\
            >\
          >
        call( BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS, ~))
        {
            typedef typename boost::tuple< \
              BOOST_PP_ENUM(N,NT2_FUTURE_TEMPLATE, ~) > \
              result_type;

            typedef typename details::tbb_future<result_type> future;

            future future_res;

            future_res.res_  = boost::make_shared< result_type > \
                ( boost::make_tuple(BOOST_PP_ENUM_PARAMS(N,a)) );

// node_type * c = new node_type( *future_res.getWork(),
//   details::tbb_task_wrapper0<details::empty_functor,future>
//     (details::empty_functor(), future_res)
//   );

            node_type * c = new node_type( *future_res.getWork(),
              details::empty_body()
              );

            future_res.getTaskQueue()->push_back(c);

            BOOST_PP_REPEAT(N, NT2_FUTURE_FORWARD_ARGS1, ~)

            future_res.attach_task(c);

            return future_res;
         }

#undef NT2_FUTURE_TEMPLATE
#undef NT2_FUTURE_FORWARD_ARGS
#undef NT2_FUTURE_FORWARD_ARGS1

#endif
