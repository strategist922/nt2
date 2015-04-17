//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_CONTAINER_DSL_SIZE_HPP_INCLUDED
#define NT2_CORE_CONTAINER_DSL_SIZE_HPP_INCLUDED

#include <nt2/include/functions/extent.hpp>

namespace nt2 { namespace ext
{
  //==========================================================================
  /*!
   * This extension point computes the size of an expression before storing it
   * in nt2::container::expression.
   *
   * \tparam Tag    Top most tag of the expression
   * \tparam Domain Domain of the expression
   * \tparam Arity  Number of children of the expression
   * \tparam Expr   The expression itself
   *
   * \return a fusion sequence containing the logical size of the expression
   *
  **/
  //==========================================================================
  template<class Tag, class Domain, int Arity, class Expr> struct size_of;
} }

#endif
