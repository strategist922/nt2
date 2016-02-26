//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2011 - 2012   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_GLOBALNONE_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_GLOBALNONE_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/include/functions/none.hpp>
#include <nt2/include/functions/colvect.hpp>

namespace nt2
{
  /*!
    @brief Checks that none element of an expression is non-zero

    @par Semantic

    For any table expression @c t:

    @code
    logical<T> r = globalnone(t);
    @endcode

    is equivalent to:

    @code
    logical<T> r = none(t(_));
    @endcode

    @see @funcref{colon}, @funcref{none}
    @param a0 Table to process

    @return An expression eventually evaluated to the result
  **/
  template<typename Args>
  BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE globalnone(Args const& a0)
  BOOST_AUTO_DECLTYPE_BODY( nt2::none(nt2::colvect(a0)) );
}

#endif
