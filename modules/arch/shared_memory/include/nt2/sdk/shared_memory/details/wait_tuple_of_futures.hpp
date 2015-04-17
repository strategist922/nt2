//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_WAIT_TUPLE_OF_FUTURES_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_WAIT_TUPLE_OF_FUTURES_HPP_INCLUDED

namespace nt2
{
  namespace details
  {

    template<std::size_t N>
    struct wait_tuple_of_futures
    {
      template <typename Tuple>
      static void call(Tuple && a)
      {

        std::get<N-1>(a).wait();
        wait_tuple_of_futures<N-1>().call(a);
      }
    };

    template<>
    struct wait_tuple_of_futures<1ul>
    {
      template <typename Tuple>
      static void call(Tuple && a)
      {
        std::get<0>(a).wait();
      }
    };

  }
}
#endif
