//==============================================================================
//         Copyright 2016  LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 20116 NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SIGNAL_FUNCTIONS_IFFT_HPP_INCLUDED
#define NT2_SIGNAL_FUNCTIONS_IFFT_HPP_INCLUDED

#include <nt2/include/functor.hpp>
#include <nt2/sdk/meta/size_as.hpp>
#include <nt2/sdk/meta/value_as.hpp>
#include <nt2/sdk/meta/tieable_hierarchy.hpp>
#include <nt2/core/container/dsl/size.hpp>
#include <nt2/core/container/dsl/value_type.hpp>

namespace nt2
{
  namespace tag
  {
    /// @brief Tag for ifft function
    struct ifft_ : ext::unspecified_<ifft_>
    {
      typedef ext::unspecified_<ifft_> parent;
      template<class... Args>
      static BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE dispatch(Args&&... args)
      BOOST_AUTO_DECLTYPE_BODY( dispatching_ifft_( ext::adl_helper(), static_cast<Args&&>(args)... ) )
    };
  }

  namespace ext
  {
    template<class Site>
    BOOST_FORCEINLINE generic_dispatcher<tag::ifft_, Site>
    dispatching_ifft_(adl_helper, boost::dispatch::meta::unknown_<Site>, ...)
    {
      return generic_dispatcher<tag::ifft_, Site>();
    }
    template<class... Args>
    struct impl_ifft_;
  }
 /*!
    @brief fft

    Computes the discrete Fourier transform of A using a Fast Fourier
    Transform (FFT) algorithm.

    @param a0 The data to calculate the fft of
    @param a1 Optional scalar specifying the number of arguements to use

    @return A matrix of size A containing \f$fft(a0)\f$
  **/

  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION(tag::ifft_, ifft, 1)

  /// INTERNAL
  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION_TPL(tag::ifft_, ifft, (A0 const&)(A1&), 2);
}

namespace nt2 { namespace ext
{
  /// INTERNAL ONLY
  template<typename Domain, typename Expr,  int N>
  struct size_of<tag::ifft_, Domain, N, Expr> : meta::size_as<Expr,0>
  {};

  /// INTERNAL ONLY
  template <typename Domain, typename Expr,  int N>
  struct value_type<tag::ifft_, Domain, N, Expr> : meta::value_as<Expr,0>
  {};
} }

#endif
