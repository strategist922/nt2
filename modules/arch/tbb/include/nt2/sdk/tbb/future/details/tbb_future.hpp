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
#include <nt2/sdk/tbb/future/details/tbb_task_wrapper.hpp>


namespace nt2
{
  namespace tag
  {
    template<class T> struct tbb_;
  }
  namespace details
  {

   template<typename result_type>
   struct tbb_future
   {
      typedef typename tbb::flow::continue_node<\
      tbb::flow::continue_msg> node_type;

      tbb_future() : work_(NULL),node_list_(NULL),node_(NULL),ready_(NULL)
      {}

       void attach_task(tbb::flow::graph * work,
                        std::vector<node_type *> * node_list,
                        node_type * node,
                        bool * ready
                        )
       {
           work_ = work;
           node_list_ = node_list;
           node_ = node;
           ready_ = ready;
       }

      void wait()
      {
            node_list_->front()->try_put(tbb::flow::continue_msg());
            work_->wait_for_all();

            delete(work_);

            for (std::size_t i =0; i<node_list_->size(); i++)
            {
              delete((*node_list_)[i]);
            }

            delete (node_list_);

            *ready_ = true;
      }

      result_type get()
      {
        if(!(*ready_)) wait();
        return res_;
      }

      template<typename F>
      tbb_future<typename boost::result_of<F>::type>
      then(F& f)
      {
        typedef typename boost::result_of<F>::type result_type;

        details::tbb_future<result_type> then_future;

        node_type * c = new node_type
          ( *work_,
            details::tbb_task_wrapper0<F,result_type>
            (f,then_future.res_)
          );

        node_list_->push_back(c);

        tbb::flow::make_edge(*node_,*c);

        then_future.attach_task(work_,node_list_,c,ready_);

        return then_future;
       }

      result_type res_;

      tbb::flow::graph * work_;
      std::vector< node_type *> * node_list_;
      node_type * node_;
      bool * ready_;
    };
   }
}

 #endif
#endif
