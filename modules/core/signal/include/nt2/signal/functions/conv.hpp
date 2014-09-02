//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SIGNAL_FUNCTIONS_CONV_HPP_INCLUDED
#define NT2_SIGNAL_FUNCTIONS_CONV_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/core/container/dsl/size.hpp>
#include <nt2/core/container/dsl/value_type.hpp>
#include <nt2/sdk/meta/value_as.hpp>
#include <nt2/sdk/meta/boxed_size.hpp>
#include <nt2/signal/options.hpp>

namespace nt2
{
  namespace tag
  {
   /*!
     @brief conv generic tag

     Represents the conv function in generic contexts.

     @par Models:
        Hierarchy
   **/
    struct conv_ : ext::unspecified_<conv_>
    {
      /// @brief Parent hierarchy
      typedef ext::unspecified_<conv_> parent;
    };
  }

  /*!

   **/
  NT2_FUNCTION_IMPLEMENTATION(tag::conv_, conv, 3)

  /// @overload
  template<typename A0, typename A1>
  typename  boost::dispatch::meta
                 ::call < tag::conv_( A0 const&, A1 const&
                                    , policy<ext::full_> const&
                                    )
                        >::type
  conv(A0 const& a0, A1 const& a1)
  {
    return conv(a0,a1,nt2::full_);
  }
}

namespace nt2 { namespace ext
{
  template<class Domain, int N, class Expr>
  struct  size_of<tag::conv_,Domain,N,Expr>
        : meta::boxed_size<Expr,3>
  {};

  template<class Domain, int N, class Expr>
  struct  value_type<tag::conv_,Domain,N,Expr>
        : meta::value_as<Expr,0>
  {};
} }

#endif
