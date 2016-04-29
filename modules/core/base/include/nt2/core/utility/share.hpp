//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_UTILITY_SHARE_HPP_INCLUDED
#define NT2_CORE_UTILITY_SHARE_HPP_INCLUDED

#include <iterator>
#include <boost/array.hpp>
#include <nt2/sdk/memory/fixed_allocator.hpp>

namespace nt2
{
  /**
    @brief Range adaptor for sharing data within Container

    Convert an ContiguousIterator range into a type suitable to initializing a
    container using the nt2::shared_ settings.

    @param begin Iterator to the first element of the range to adapt
    @param end   Iterator to the past-the-end element of the range to adapt

    @return A nt2::memory::fixed_allocator adapting the initial Range.
  */
  template<class ContiguousIterator>
  BOOST_FORCEINLINE
  memory::fixed_allocator < typename  std::iterator_traits
                                      <ContiguousIterator>::value_type
                          >
  share(ContiguousIterator begin, ContiguousIterator end)
  {
    typedef typename std::iterator_traits<ContiguousIterator>::value_type v_t;
    memory::fixed_allocator<v_t> that(begin,end);
    return that;
  }

  /**
    @brief Boost array adaptor for sharing data within Container

    Convert an array range into a type suitable to initializing a
    container using the nt2::shared_ settings.

    @param values Boost array to adapt

    @return A nt2::memory::fixed_allocator adapting the initial Range.
  */
  template<class Type, std::size_t Size>
  BOOST_FORCEINLINE memory::fixed_allocator<Type>
  share( boost::array<Type,Size>& values)
  {
    memory::fixed_allocator<Type> that(values.begin(),values.end());
    return that;
  }

  /**
    @brief C array adaptor for sharing data within Container

    Convert an array range into a type suitable to initializing a
    container using the nt2::shared_ settings.

    @param values C array to adapt

    @return A nt2::memory::fixed_allocator adapting the initial Range.
  */
  template<class Type, std::size_t Size>
  BOOST_FORCEINLINE memory::fixed_allocator<Type>
  share( Type (&values)[Size])
  {
    memory::fixed_allocator<Type> that(&values[0], &values[0]+Size);
    return that;
  }
}

#endif
