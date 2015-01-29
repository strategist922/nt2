//==============================================================================
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2015   NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_MEMORY_DETAILS_BUFFER_HPP_INCLUDED
#define NT2_SDK_MEMORY_DETAILS_BUFFER_HPP_INCLUDED

#include <boost/dispatch/meta/select.hpp>
#include <type_traits>
#include <iterator>
#include <cstddef>
#include <memory>

namespace nt2 { namespace details
{
  // default construct (if needed) a range
  template<typename Iterator, typename Allocator>
  BOOST_FORCEINLINE void may_construct(Iterator b, Iterator e, Allocator& a)
  {
    using alloc_t   = std::allocator_traits<Allocator>;
    using type      = typename std::iterator_traits<Iterator>::value_type;
    using reference = typename std::iterator_traits<Iterator>::reference;

     std::for_each( b, e, boost::select<std::is_trivial<type>>
                          ( [](reference) {}
                          , [&a](reference x) { alloc_t::construct(a,&x); }
                          )
                  );
  }

  // call destructor (if needed) on a range
  template<typename Iterator, typename Allocator>
  BOOST_FORCEINLINE void may_destroy(Iterator b, Iterator e, Allocator& a)
  {
    using alloc_t   = std::allocator_traits<Allocator>;
    using type      = typename std::iterator_traits<Iterator>::value_type;
    using reference = typename std::iterator_traits<Iterator>::reference;

    std::for_each ( b, e, boost::select<std::is_trivial<type>>
                          ( [](reference) {}
                          , [&a](reference x) { alloc_t::destroy(a,&x); }
                          )
                  );
  }

  // Allocator deleter for unique_ptr
  template<typename Allocator> struct deleter
  {
    using alloc_t = std::allocator_traits<Allocator>;
    using pointer = typename alloc_t::pointer;

    void operator()(pointer p) const
    {
      details::may_destroy(p , p+sz, *pa);
      alloc_t::deallocate(*pa,p,cp);
    };

    Allocator*  pa;
    std::size_t sz,cp;
  };
} }

#endif
