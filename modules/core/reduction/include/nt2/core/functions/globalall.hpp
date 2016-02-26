//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2011 - 2012   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_GLOBALALL_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_GLOBALALL_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/include/functions/all.hpp>
#include <nt2/include/functions/global.hpp>

namespace nt2
{
  /*!
    @brief Checks that all elements of an expression is non-zero

    @par Semantic

    For any table expression @c t:

    @code
    logical<T> r = globalall(t);
    @endcode

    is equivalent to:

    @code
    logical<T> r = all(t(_))(1);
    @endcode

    @see @funcref{colon}, @funcref{all}
    @param a0 Table to process

    @return An expression eventually evaluated to the result
  **/
  template<typename Args>
  BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE globalall(Args const& a0)
  BOOST_AUTO_DECLTYPE_BODY( global(nt2::functor<tag::all_>(), a0) );
}
#endif
