//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_OPENMP_FUTURE_DETAILS_OPENMP_FUTURE_HPP_INCLUDED
#define NT2_SDK_OPENMP_FUTURE_DETAILS_OPENMP_FUTURE_HPP_INCLUDED

#if defined(_OPENMP) && _OPENMP >= 201307 /* OpenMP 4.0 */

#include <omp.h>

#include <boost/move/move.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace nt2
{
  namespace tag
  {
    template<class T> struct openmp_;
  }

    namespace details
    {
        class openmp_future_base
        {
        protected:

            openmp_future_base () {}
            ~openmp_future_base () {}

        public:

            static bool getGraphIsCompleted ()
            {
                if (NULL == graph_is_completed_)
                {
                    graph_is_completed_ = new bool(false);
                    printf("Create new isready with value %d\n",*graph_is_completed_);
                }
                return *graph_is_completed_;
            }

            static void kill_graph ()
            {
                if (NULL != graph_is_completed_)
                {
                    delete graph_is_completed_;
                    graph_is_completed_ = NULL;
                }
            }

        private:

            static bool * graph_is_completed_;
        };

  bool * openmp_future_base::graph_is_completed_ = NULL;

  template<typename result_type>
  struct openmp_future : openmp_future_base
  {
      openmp_future() : res_(new result_type),ready_(new bool(false))
      {}

      template<typename previous_future>
      void attach_previous_future(previous_future const & pfuture)
      {
          pfuture_ = boost::make_shared<previous_future> (pfuture);
      }

      bool is_ready() const
      {
          return *ready_;
      }

      void wait()
      {
          if(!getGraphIsCompleted())
          {
              #pragma omp taskwait
              kill_graph();
          }
      }

      result_type get()
      {
          if(!is_ready()) wait();
          return *res_;
      }

      template<typename F>
      openmp_future<typename boost::result_of<F(result_type)>::type>
      then(BOOST_FWD_REF(F) f)
      {
          typedef typename boost::result_of<F>::type then_result_type;

          details::openmp_future<then_result_type> then_future;

          then_future.attach_previous_future(*this);

          result_type & prev( *res_ );
          then_result_type & next( *(then_future.res_) );

          #pragma omp task shared(f,then_future) depend(in: prev) depend(out: next)
          {
              next = f(then_future);
              *(then_future.ready_) = true;
          }

          return then_future;
      }

      boost::shared_ptr<void> pfuture_;
      boost::shared_ptr<result_type> res_;
      boost::shared_ptr<bool> ready_;

   };
  }
 }

 #endif
#endif
