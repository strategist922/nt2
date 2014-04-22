//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef FILTERS_REDUCTION_HPP_INCLUDED
#define FILTERS_REDUCTION_HPP_INCLUDED

#include <iostream>
#include <cstddef>
#include "window.hpp"
#include "reduction.hpp"
#include <boost/simd/sdk/meta/cardinal_of.hpp>

namespace nt2 { namespace details
{
  // Perform fold operation on a single column
  template<typename Operation, typename Type, typename Data, typename Source>
  BOOST_FORCEINLINE
  void reduce_column(std::size_t j, std::size_t i, Data& d, Source const& in)
  {
    typedef typename Operation::template rebind<Operation::height,1>::other h_op;

    nt2::window<h_op,Type> x(j,i,in);
    d = x.fold();
  }

  // Unroll reduction over fixed set of column
  template<std::size_t Size, typename Type, std::size_t Iter = 0>
  struct reducer_
  {
    template<typename Window, typename Source> BOOST_FORCEINLINE
    static void call(std::size_t j, Window& w, Source const& in)
    {
      static const std::size_t card = boost::simd::meta::cardinal_of<Type>::value;
      reduce_column<typename Window::operation_type,Type>(j, Iter*card, w[Iter], in);
      reducer_<Size,Type,Iter+1>::call(j,w,in);
    }
  };

  template<std::size_t Size, typename Type>
  struct reducer_<Size,Type,Size>
  {
    template<typename Window, typename Source> BOOST_FORCEINLINE
    static void call(std::size_t, Window&, Source const&) {}
  };
} }

#endif
