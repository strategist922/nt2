//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txtq
//==============================================================================
#ifndef NT2_SDK_TBB_THREAD_UTILITY_HPP_INCLUDED
#define NT2_SDK_TBB_THREAD_UTILITY_HPP_INCLUDED

#if defined(NT2_USE_TBB)

#include <omp.h>
#include <tbb/task_scheduler_init.h>
#include <nt2/sdk/tbb/tbb_initializer.hpp>
#include <nt2/sdk/shared_memory/details/thread_utility.hpp>
#include <iostream>

namespace nt2
{
  namespace tag
  {
      template<class T> struct tbb_;
  }

   template <typename Site>
   struct get_num_threads_impl< nt2::tag::tbb_<Site> >
   {
     inline int call() const
     {
       return tbb::task_scheduler_init::default_num_threads();
     }
   };

   template <typename Site>
   struct set_num_threads_impl< nt2::tag::tbb_<Site> >
   {
     inline void call(int n) const
     {
       nt2::tbb_initializer().init(n);
     }
   };
}

#endif
#endif
