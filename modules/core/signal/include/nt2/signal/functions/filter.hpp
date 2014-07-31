//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_FILTER_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_FILTER_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/sdk/meta/value_as.hpp>
#include <nt2/core/container/dsl/size.hpp>
#include <nt2/core/container/dsl/value_type.hpp>
#include <nt2/sdk/meta/tieable_hierarchy.hpp>
#include <nt2/include/functions/size.hpp>

namespace nt2
{
  namespace tag
  {
   /*!
     @brief filter generic tag

     Represents the filter function in generic contexts.

     @par Models:
        Hierarchy
   **/
    struct filter_ : ext::tieable_<filter_>
    {
      /// @brief Parent hierarchy
      typedef ext::tieable_<filter_> parent;
    };
  }
  /*!
   * Apply filter algorithm to find local minimum of a function
   *
   * \param f        Function to optimize
   * \param tspan    Table of times at which to return the state of the system
   * \param t0       Start time
   * \param t1       Emd time
   * \param init     Initial value
   * \param f        Function to process output
   *
   * \return  a tuple containing the time points and the solution at the time given in the first return value
   */
  NT2_FUNCTION_IMPLEMENTATION(tag::filter_, filter, 4)
}

namespace nt2 { namespace ext
{
  template<class Domain, int N, class Expr>
  struct  size_of<tag::filter_,Domain,N,Expr>
  {
    typedef _0D result_type;

    BOOST_FORCEINLINE result_type operator()(Expr& e) const
    {
      return _0D();
    }
  };

  template<class Domain, int N, class Expr>
  struct  value_type<tag::filter_,Domain,N,Expr>
        : meta::value_as<Expr,2>  //this needs to be the size of the Expr
  {};

} }

#endif
