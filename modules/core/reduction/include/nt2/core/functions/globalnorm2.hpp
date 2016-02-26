//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2011 - 2012   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_GLOBALNORM2_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_GLOBALNORM2_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/include/functions/norm2.hpp>
#include <nt2/include/functions/colvect.hpp>

namespace nt2
{
  /*!
    @brief euclidian norm of a whole table expression elements

    @par Semantic

    For any table expression of T @c t integer or weights w   and any integer @c n:

    @code
    T r = globalnorm2(t);
    @endcode

    is equivalent to:

    if w is an integer

    @code
    T r = norm2(t(_))(1);
    @endcode

    @par Note:
    n default to firstnonsingleton(t)

    @par alias:
    norm_eucl

    @see @funcref{firstnonsingleton}, @funcref{norm2}
    @param a0 Table to process

    @return An expression eventually evaluated to the result
  **/
  template<typename Args>
  BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE globalnorm2(Args const& a0)
  BOOST_AUTO_DECLTYPE_BODY( nt2::norm2(nt2::colvect(a0)) );
}

#endif
