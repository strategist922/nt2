//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2011 - 2012   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_GLOBALANY_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_GLOBALANY_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/include/functions/any.hpp>
#include <nt2/include/functions/global.hpp>

namespace nt2
{
  /*!
    @brief Checks that any elements of an expression is non-zero

    @par Semantic

    For any table expression @c t:

    @code
    logical<T> r = globalany(t);
    @endcode

    is equivalent to:

    @code
    logical<T> r = any(t(_))(1);
    @endcode

    @see @funcref{colon}, @funcref{any}
    @param a0 expression table to process

    @return An expression eventually evaluated to the result
  **/
  template<typename Args>
  BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE globalany(Args const& a0)
  BOOST_AUTO_DECLTYPE_BODY( global(nt2::functor<tag::any_>(), a0) );
}

#endif
