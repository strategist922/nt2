//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_TBB_FUTURE_DETAILS_TBB_FUTURE_HPP_INCLUDED
#define NT2_SDK_TBB_FUTURE_DETAILS_TBB_FUTURE_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <tbb/tbb.h>
#include <tbb/flow_graph.h>

#include <vector>
#include <cstdio>

#include <boost/move/utility.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <nt2/sdk/tbb/future/details/tbb_task_wrapper.hpp>

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

            static tbb::flow::graph *getWork ()
            {
                if (NULL == nt2_graph_)
                {
                   printf("Create new graph\n");
                   nt2_graph_ = new tbb::flow::graph;
                }

                return nt2_graph_;
            }

            static tbb::flow::broadcast_node
            <tbb::flow::continue_msg> *getStart ()
            {
                if (NULL == start_task_)
                {
                    printf("Create new start task\n");
                    start_task_ =
                    new tbb::flow::broadcast_node
                    <tbb::flow::continue_msg>(*getWork());
                }
                return (start_task_);
            }

            static std::vector< \
            tbb::flow::continue_node<  \
            tbb::flow::continue_msg> * \
            > * getTaskQueue ()
            {
                if (NULL == task_queue_)
                {
                    printf("Create new task queue\n");

                    task_queue_ = new std::vector< \
                    tbb::flow::continue_node< \
                    tbb::flow::continue_msg> * \
                    >;

                    task_queue_->reserve(100);
                }
                return task_queue_;
            }

            static boost::shared_ptr<bool> getGraphIsCompleted ()
            {
                if (NULL == graph_is_completed_)
                {
                    graph_is_completed_ = \
                      new boost::shared_ptr<bool>(new bool(false));
                    printf("Create new isready with value %d\n",**graph_is_completed_);
                }
                return *graph_is_completed_;
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

                if (NULL != graph_is_completed_)
                {
                    delete graph_is_completed_;
                    graph_is_completed_ = NULL;
                }
            }

            private:

            static tbb::flow::graph *
              nt2_graph_;

            static tbb::flow::broadcast_node<tbb::flow::continue_msg> *
              start_task_;

            static std::vector< \
            tbb::flow::continue_node< \
            tbb::flow::continue_msg> * \
            > *
              task_queue_;

            static boost::shared_ptr<bool> *
              graph_is_completed_;
        };

        tbb::flow::graph *
        tbb_future_base::nt2_graph_ = NULL;

        tbb::flow::broadcast_node<tbb::flow::continue_msg> *
        tbb_future_base::start_task_ = NULL;

        std::vector< \
        tbb::flow::continue_node< \
        tbb::flow::continue_msg> * \
        > *
        tbb_future_base::task_queue_ = NULL;

        boost::shared_ptr<bool> *
        tbb_future_base::graph_is_completed_ = NULL;

        template<typename result_type>
        struct tbb_future : public tbb_future_base
        {
            typedef typename tbb::flow::continue_node<\
            tbb::flow::continue_msg> node_type;

            tbb_future() : node_(NULL),res_(new result_type)
            {}

            template< typename previous_future>
            void attach_previous_future(previous_future const & pfuture)
            {
                pfuture_ = boost::make_shared<previous_future> (pfuture);
            }


            void attach_task(node_type * node)
            {
                node_ = node;
                ready_ = getGraphIsCompleted();
            }

            bool is_ready() const
            {
                return *ready_;
            }

            void wait()
            {
                if(!is_ready())
                {
                    getStart()->try_put(tbb::flow::continue_msg());
                    getWork()->wait_for_all();
                    *ready_ = true;
                    kill_graph();
                }
            }

            result_type get()
            {
                if(!is_ready()) wait();
                return *res_;
            }

            template<typename F>
            details::tbb_future<
              typename boost::result_of<F(result_type)>::type
              >
            then(BOOST_FWD_REF(F) f)
            {
                typedef typename boost::result_of<F(result_type)>::type
                  then_result_type;

                details::tbb_future<then_result_type>
                  then_future;

                then_future.attach_previous_future(*this);

                node_type * c = new node_type
                  ( *getWork(),
                      details::tbb_task_wrapper1<
                        F,
                        then_result_type,
                        result_type const &
                        >
                   (boost::forward<F>(f)
                    , *(then_future.res_)
                    , *res_ )
                   );

                getTaskQueue()->push_back(c);

                tbb::flow::make_edge(*node_,*c);

                then_future.attach_task(c);

                return then_future;
           }

            node_type * node_;
            boost::shared_ptr<result_type> res_;
            boost::shared_ptr<void> pfuture_;
            boost::shared_ptr<bool> ready_;
        };
    }
}

 #endif
#endif
