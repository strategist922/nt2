//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_TBB_FUTURE_DETAILS_TBB_SHARED_FUTURE_HPP_INCLUDED
#define NT2_SDK_TBB_FUTURE_DETAILS_TBB_SHARED_FUTURE_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <tbb/tbb.h>
#include <tbb/flow_graph.h>

#include <vector>
#include <cstdio>
#include <memory>
#include <type_traits>
#include <future>
#include <atomic>

#include <nt2/sdk/tbb/future/details/tbb_task_wrapper.hpp>
#include <nt2/sdk/tbb/future/details/tbb_future.hpp>

namespace nt2
{
  namespace tag
  {
    template<class T> struct tbb_;
  }

  namespace details
  {
    template<typename result_type>
    struct tbb_shared_future
    : public tbb_future_base
    , public std::shared_future<result_type>
    {
      typedef typename tbb::flow::continue_node<
      tbb::flow::continue_msg> node_type;

      tbb_shared_future()
      : tbb_future_base()
      , std::shared_future<result_type>()
      , node_(NULL)
      , continued_(false)
      , ready_( graph_launched_ )
      {
      }

       tbb_shared_future( std::shared_future<result_type> && other)
      : tbb_future_base()
      , std::shared_future<result_type>(
        std::forward< std::shared_future<result_type> >(other)
        )
      , node_(NULL)
      , continued_(false)
      , ready_( graph_launched_ )
      {
      }

      void wait()
      {
        if( ! ready_->test_and_set(std::memory_order_acquire) )
        {
          getStart()->try_put(tbb::flow::continue_msg());
          getWork()->wait_for_all();
          kill_graph();
        }

      }

      result_type get()
      {
        if(!continued_)
          wait();

        std::shared_future<result_type> & tmp(*this);
        return tmp.get();
      }

      template<typename F>
      details::tbb_future<
      typename std::result_of<F(tbb_shared_future)>::type
      >
      then(F&& f)
      {
        typedef typename std::result_of<F(tbb_shared_future)>::type
        then_result_type;

        typedef typename details::tbb_shared_future<then_result_type>
        then_future_type;

        node_type * node = node_;
        continued_ = true;

        details::tbb_task_wrapper<
        F,
        then_result_type,
        tbb_shared_future
        >
        packaged_task
        (std::forward<F>(f)
        ,tbb_shared_future(*this)
        );

        then_future_type then_future( packaged_task.get_future() );

        node_type * c
        = new node_type
        ( *( then_future.getWork() )
        , std::move(packaged_task)
        );

        tbb::flow::make_edge(*node,*c);
        then_future.getTaskQueue()->push_back(c);
        then_future.attach_task(c);

        return then_future;
      }


      void attach_task(node_type * node)
      {
        node_ = node;
      }

// own members
      node_type * node_;
      bool continued_;
      std::shared_ptr< std::atomic_flag > ready_;
    };
  }
}

#endif
#endif
