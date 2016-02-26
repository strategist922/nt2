//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2011 - 2012   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_GLOBALMEANAD_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_GLOBALMEANAD_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/include/functions/meanad.hpp>
#include <nt2/include/functions/colvect.hpp>

namespace nt2
{
  /*!
    @brief Mean of the absolute deviation of all the elements of an expression

    Computes  the absolute deviation of all the elements of a table expression

    @par Semantic

    For any table expression :

    @code
    T r = globalmeanad(t);
    @endcode

    is equivalent to:

    @code
    T r = meanad(t(_))(1);
    @endcode


    @see @funcref{colon}, @funcref{meanad}
    @param a0 Table to process

    @return An expression eventually evaluated to the result
  **/
  template<typename Args>
  BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE globalmeanad(Args const& a0)
  BOOST_AUTO_DECLTYPE_BODY( nt2::meanad(nt2::colvect(a0)) );
}

#endif
