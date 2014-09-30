//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txtq
//==============================================================================
#ifndef NT2_SDK_HPX_THREAD_UTILITY_HPP_INCLUDED
#define NT2_SDK_HPX_THREAD_UTILITY_HPP_INCLUDED

#if defined(NT2_USE_HPX)

#include <hpx/hpx_fwd.hpp>
#include <nt2/sdk/shared_memory/details/thread_utility.hpp>
#include <cstdio>

namespace nt2
{
  namespace tag
  {
      template<class T> struct hpx_;
  }

   template <typename Site>
   struct get_num_threads_impl< nt2::tag::hpx_<Site> >
   {
     inline int call() const
     {
      return hpx::get_os_thread_count();
     }
   };

   template <typename Site>
   struct set_num_threads_impl< nt2::tag::hpx_<Site> >
   {
     inline void call(int n) const
     {
        printf("HPX cannot set the number of OS-threads online\n");
     }
   };

   template <typename Site>
   struct get_thread_id_impl< nt2::tag::hpx_<Site> >
   {
     inline int call() const
     {
       return hpx::get_worker_thread_num();
     }
   };
}

#endif
#endif
