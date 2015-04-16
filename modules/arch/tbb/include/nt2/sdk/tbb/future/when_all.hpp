//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_TBB_FUTURE_WHEN_ALL_HPP_INCLUDED
#define NT2_SDK_TBB_FUTURE_WHEN_ALL_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <tbb/tbb.h>
#include <tbb/flow_graph.h>

#include <nt2/sdk/shared_memory/future.hpp>
#include <nt2/sdk/tbb/future/details/tbb_future.hpp>
#include <nt2/sdk/tbb/future/details/tbb_shared_future.hpp>

#include <cstdio>
#include <initializer_list>

namespace nt2
{

  namespace details
  {
    template <class Future, typename Node_raw, typename A>
    static void link_node(Future & f, Node_raw & c, A & a)
    {
      tbb::flow::make_edge( *(a.node_)
                          , *c
                          );
      f.attach_previous_future( a );
    }
  }

  template<class Site>
  struct when_all_impl< tag::tbb_<Site> >
  {
    typedef typename tbb::flow::continue_node<
    tbb::flow::continue_msg> node_type;

    template <typename Future>
    details::tbb_future< std::vector<Future> >
    static call( std::vector<Future> & lazy_values )
    {
      typedef typename std::vector<Future> whenall_vector;
      typedef typename details::tbb_future< whenall_vector >
      whenall_future;

      details::tbb_task_wrapper<
        std::function< whenall_vector() >
      , whenall_vector
      >
      packaged_task( [&](){ return lazy_values; } );

      whenall_future future_res( packaged_task.get_future() );

      node_type * c = new node_type( *(future_res.getWork())
                                   , std::move(packaged_task)
                                   );

      future_res.getTaskQueue()->push_back(c);

      for (std::size_t i=0; i<lazy_values.size(); i++)
      {
        tbb::flow::make_edge(*(lazy_values[i].node_),*c);
      }

      future_res.attach_task(c);
      return future_res;
    }

    template< typename ... A >
    typename details::tbb_future<
    std::tuple< details::tbb_shared_future<A> ... >
    >
    static call( details::tbb_future<A> & ...a )
    {
      typedef typename std::tuple< details::tbb_shared_future<A> ... >
      whenall_tuple;

      typedef typename details::tbb_future< whenall_tuple >
      whenall_future;

      details::tbb_task_wrapper<
        std::function<whenall_tuple()>
      , whenall_tuple
      >
      packaged_task( [&]()
                     { return std::make_tuple<
                          details::tbb_shared_future<A> ...
                          >
                       ( a.share()... );
                     }
                   );

      whenall_future future_res (packaged_task.get_future());

      node_type * c
      = new node_type( *future_res.getWork(), std::move(packaged_task) );

      future_res.getTaskQueue()->push_back(c);
      future_res.attach_task(c);

      // Some trick to call link_node multiple times
      return (void)std::initializer_list<int>
      { ( static_cast<void>( details::link_node(future_res, c, a) )
        , 0
        )...
      }
      , future_res;
    }
  };
}

#endif
#endif
