//==============================================================================
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_SELECT_HPP_INCLUDED
#define BOOST_DISPATCH_META_SELECT_HPP_INCLUDED

#include <type_traits>

namespace boost
{
  template<typename Cond, typename FT, typename FF>
  typename std::enable_if<Cond::value,FT const&>::type
  select(FT const& f, FF const&)
  {
    return f;
  }

  template<typename Cond, typename FT, typename FF>
  typename std::enable_if<!Cond::value,FF const&>::type
  select(FT const&, FF const& f)
  {
    return f;
  }
}

#endif
