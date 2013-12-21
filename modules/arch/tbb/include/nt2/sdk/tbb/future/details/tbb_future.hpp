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

namespace nt2
{
  namespace tag
  {
    template<class T> struct tbb_;
  }

  namespace details
  {
    template<typename F>
    struct tbb_continuation: public tbb::task
    {

      tbb_continuation(F & f) : f_(f)
      {}

      tbb::task* execute()
      {
        f_();
        return NULL;
      }

      private:
        F & f_;
    };

    template<typename result_type>
    struct tbb_future
    {
      tbb_future() : work_(NULL)
      {}

      void attach_task(tbb::task * work)
      {
        work_ = work;
      }

      result_type get()
      {
        if( work_ == NULL ) return res_;
        else work_->wait_for_all();
        return res_;
      }

      void wait()
      {
        if( work_!= NULL )
        work_->wait_for_all();
      }

      template<typename F>
      tbb_future< \
        typename boost::result_of<F>::type\
        > then(F& f)
      {
        details::tbb_future< \
          typename boost::result_of<F>::type > \
          then_future;

        tbb_continuation<F>* c =
          new( work_->allocate_continuation() ) \
             tbb_continuation<F>(f);

        then_future.attach_task(c);

        return then_future;
      }

     private:
      tbb::task * work_;
     public:
      result_type res_;
    };
   }
 }

 #endif
#endif
