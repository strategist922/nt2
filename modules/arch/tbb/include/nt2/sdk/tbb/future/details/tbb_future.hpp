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
   template<typename result_type>
   struct tbb_future
   {
      tbb_future() : work_(NULL)
      {}

      void attach_task(tbb::task_group * work)
      {
        work_ = work;
      }

      void wait()
      {
          if( work_!= NULL )
          { work_->wait();
            delete(work_);
            work_ = NULL;
          }
      }

      result_type get()
      {
        if( work_ != NULL ) wait();
        return res_;
      }

      result_type res_;

     private:
      tbb::task_group * work_;

    };
   }
}

 #endif
#endif
