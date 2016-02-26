//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2011 - 2012   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_GLOBALASUM1_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_GLOBALASUM1_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/include/functions/asum1.hpp>
#include <nt2/include/functions/colvect.hpp>

namespace nt2
{
  /*!
    @brief Sum of the absolute values of all the elements of a table expression

    Computes the sum of  the absolute value of all the elements
    of a table expression

    @par Semantic

    For any table @c t:

    @code
    T r = globalasum1(t);
    @endcode

    is equivalent to:

    @code
    T r = asum1(a(_))(1);
    @endcode

    @see @funcref{colon}, @funcref{asum1}
    @param a0 Table expression to process

    @return An expression eventually evaluated to the result
  **/
  template<typename Args>
  BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE globalasum1(Args const& a0)
  BOOST_AUTO_DECLTYPE_BODY( nt2::asum1(nt2::colvect(a0)) );
}

#endif
