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
#ifndef NT2_SDK_TBB_FUTURE_FUTURE_HPP_INCLUDED
#define NT2_SDK_TBB_FUTURE_FUTURE_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <tbb/tbb.h>

#include <vector>

#include <boost/move/move.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/iteration/iterate.hpp>

#include <nt2/sdk/shared_memory/future.hpp>
#include <nt2/sdk/tbb/future/details/tbb_future.hpp>
#include <nt2/sdk/tbb/future/details/tbb_task_wrapper.hpp>

namespace nt2
{
  namespace tag
  {
    template<class T> struct tbb_;
  }

  template<class Site, class result_type>
  struct make_future<tag::tbb_<Site> , result_type>
  {
     typedef nt2::details::tbb_future<result_type> type;
  };

  template<class Site>
  struct async_impl< tag::tbb_<Site> >
  {
    typedef typename tbb::flow::continue_node<\
    tbb::flow::continue_msg> node_type;

#define BOOST_PP_ITERATION_PARAMS_1 (3, \
( 0, BOOST_DISPATCH_MAX_ARITY, "nt2/sdk/tbb/future/future.hpp") \
)
#include BOOST_PP_ITERATE()
  };
}

#endif
#endif

#else

#define N BOOST_PP_ITERATION()

#define NT2_FUTURE_FORWARD_ARGS(z,n,t) BOOST_FWD_REF(A##n) a##n
#define NT2_FUTURE_FORWARD_ARGS2(z,n,t) boost::forward<A##n>(a##n)

  template< typename F\
            BOOST_PP_COMMA_IF(N)\
            BOOST_PP_ENUM_PARAMS(N, typename A) >
  inline typename make_future< tag::tbb_<Site>,\
            typename boost::result_of<\
            F(BOOST_PP_ENUM_PARAMS(N, A))>::type \
            >::type
  call(F & f\
       BOOST_PP_COMMA_IF(N)\
       BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS, ~)\
      )
  {
    details::tbb_future<typename boost::result_of<\
                  F(BOOST_PP_ENUM_PARAMS(N, A))\
                  >::type> future_res;

    tbb::flow::graph * work = new tbb::flow::graph;

    std::vector<node_type> * node_list = new std::vector<node_type>;

    node_list->push_back(\
       node_type( *work,\
       BOOST_PP_CAT(details::tbb_task_wrapper,N)\
       <F,typename boost::result_of<\
       F(BOOST_PP_ENUM_PARAMS(N, A))\
       >::type\
       BOOST_PP_COMMA_IF(N)\
       BOOST_PP_ENUM_PARAMS(N,A) \
       >\
       (f, future_res.res_\
       BOOST_PP_COMMA_IF(N)\
       BOOST_PP_ENUM(N,NT2_FUTURE_FORWARD_ARGS2, ~)\
       ) );

    future_res.attach_task(work,node_list,&node_list->begin());

    return future_res;
  }

#undef NT2_FUTURE_FORWARD_ARGS
#undef NT2_FUTURE_FORWARD_ARGS2

#endif
