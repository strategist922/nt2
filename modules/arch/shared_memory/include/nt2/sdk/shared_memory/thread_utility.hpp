//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txtq
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_THREAD_UTILITY_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_THREAD_UTILITY_HPP_INCLUDED

#include <nt2/sdk/functor/site.hpp>
#include <nt2/sdk/shared_memory.hpp>

namespace nt2
{
  inline int get_num_threads()
  {
    typedef typename boost::dispatch::default_site<void>::type Arch;

    return nt2::get_num_threads_impl<Arch>().call();
  }

  inline void set_num_threads(int const & n)
  {
    typedef typename boost::dispatch::default_site<void>::type Arch;

    nt2::set_num_threads_impl<Arch>().call( n );
  }

 }

#endif
