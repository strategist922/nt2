//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef FILTERS_WINDOW_HPP_INCLUDED
#define FILTERS_WINDOW_HPP_INCLUDED

#include <cstddef>
#include <boost/dispatch/attributes.hpp>
#include "window_helpers.hpp"

namespace nt2
{
  // Windows of data to process
  template<typename Operation, typename T>
  struct window : boost::array<T,Operation::height*Operation::width>
  {
    typedef Operation operation_type;
    typedef boost::array<T,Operation::height*Operation::width> parent;

    BOOST_FORCEINLINE window() {}

    template<typename Source>
    BOOST_FORCEINLINE window(std::size_t j, std::size_t i, Source const& in)
    {
      nt2::details::extract_window<Operation,T>::call(j,i, parent::c_array(),in);
    }

    BOOST_FORCEINLINE T fold() const
    {
      return nt2::details::fold_<Operation>::call(*this);
    }

    BOOST_FORCEINLINE void slide()
    {
      nt2::details::slide_<Operation,T>::call( *this );
    }

    BOOST_FORCEINLINE void shift()
    {
      nt2::details::circ_shift_<Operation::height>::call(*this);
    }
  };
}

#endif
