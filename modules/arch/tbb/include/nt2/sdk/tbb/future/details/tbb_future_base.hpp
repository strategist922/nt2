//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_TBB_FUTURE_DETAILS_TBB_FUTURE_BASE_HPP_INCLUDED
#define NT2_SDK_TBB_FUTURE_DETAILS_TBB_FUTURE_BASE_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <tbb/tbb.h>
#include <tbb/flow_graph.h>

#include <vector>
#include <memory>
#include <type_traits>
#include <atomic>

namespace nt2
{
  namespace tag
  {
    template<class T> struct tbb_;
  }

  namespace details
  {
    class tbb_future_base
    {
    protected:

      tbb_future_base () {}
      ~tbb_future_base () {}

    public:

      static tbb::flow::graph *
      getWork ()
      {
        if(NULL == nt2_graph_)
        {
          nt2_graph_ = new tbb::flow::graph;
          graph_launched_ -> clear();
        }
        return nt2_graph_;
      }

      static tbb::flow::continue_node<tbb::flow::continue_msg> *
      getStart ()
      {
        if (NULL == start_task_)
        {
          start_task_ =
          new tbb::flow::continue_node
          <tbb::flow::continue_msg>
          (* getWork()
          , []( tbb::flow::continue_msg ){}
          );
        }
        return (start_task_);
      }

      static std::vector<
      tbb::flow::continue_node< tbb::flow::continue_msg> *
      > *
      getTaskQueue ()
      {
        if (NULL == task_queue_)
        {
          task_queue_ = new std::vector<
          tbb::flow::continue_node<tbb::flow::continue_msg> *
          >;

          task_queue_->reserve(1000);
        }
        return task_queue_;
      }

      static void kill_graph ()
      {
        if (NULL != nt2_graph_)
        {
          delete nt2_graph_;
          nt2_graph_ = NULL;
        }

        if (NULL != start_task_)
        {
          delete start_task_;
          start_task_ = NULL;
        }

        if (NULL != task_queue_)
        {
          for (std::size_t i =0; i<task_queue_->size(); i++)
          {
            delete( (*task_queue_)[i] );
          }

          delete task_queue_;
          task_queue_ = NULL;
        }

        // Allocate a new flag for next graph
        graph_launched_ = std::make_shared< std::atomic_flag >();
        graph_launched_ -> clear();
      }

      static std::shared_ptr< std::atomic_flag > graph_launched_;

    private:

// static members
      static tbb::flow::graph *
      nt2_graph_;

      static tbb::flow::continue_node<tbb::flow::continue_msg> *
      start_task_;

      static std::vector<
      tbb::flow::continue_node<tbb::flow::continue_msg> *
      > *
      task_queue_;
    };

    tbb::flow::graph *
    tbb_future_base::nt2_graph_ = NULL;

    tbb::flow::continue_node<tbb::flow::continue_msg> *
    tbb_future_base::start_task_ = NULL;

    std::vector<
    tbb::flow::continue_node<tbb::flow::continue_msg> *
    > *
    tbb_future_base::task_queue_ = NULL;

    std::shared_ptr< std::atomic_flag >
    tbb_future_base::graph_launched_( std::make_shared< std::atomic_flag >()
                                    );
  }
}

#endif
#endif
