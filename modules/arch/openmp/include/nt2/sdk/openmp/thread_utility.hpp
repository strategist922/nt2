//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txtq
//==============================================================================
#ifndef NT2_SDK_OPENMP_THREAD_UTILITY_HPP_INCLUDED
#define NT2_SDK_OPENMP_THREAD_UTILITY_HPP_INCLUDED

#if defined(_OPENMP) && _OPENMP >= 200203 /* OpenMP 2.0 */

#include <omp.h>
#include <nt2/sdk/shared_memory/details/thread_utility.hpp>

namespace nt2
{
  namespace tag
  {
      template<class T> struct openmp_;
  }

   template <typename Site>
   struct get_num_threads_impl< nt2::tag::openmp_<Site> >
   {
     inline int call() const
     {
       return omp_get_max_threads();
     }
   };

   template <typename Site>
   struct set_num_threads_impl< nt2::tag::openmp_<Site> >
   {
     inline void call(int n) const
     {
       return omp_set_num_threads(n);
     }
   };
}

#endif
#endif
