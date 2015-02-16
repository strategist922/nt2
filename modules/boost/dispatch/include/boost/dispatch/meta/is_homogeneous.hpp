//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_DISPATCH_META_IS_HOMOGENEOUS_HPP_INCLUDED
#define BOOST_DISPATCH_META_IS_HOMOGENEOUS_HPP_INCLUDED

#include <boost/fusion/include/is_sequence.hpp>
#include <boost/dispatch/meta/all.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/mpl/at.hpp>

namespace boost { namespace dispatch { namespace meta
{
  /*!

  **/
  template< typename T
          , bool isSequence = boost::fusion::traits::is_sequence<T>::value
          >
  struct is_homogeneous : boost::mpl::true_
  {
  };

  template<typename T>
  struct  is_homogeneous<T,true>
        : all_seq < std::is_same< typename boost::mpl::at_c<T,0>::type
                                , boost::mpl::_
                                >
                  , T
                  >
  {};
} } }

#endif
