//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_ELLIPTIC_FUNCTIONS_ELLPE_HPP_INCLUDED
#define NT2_ELLIPTIC_FUNCTIONS_ELLPE_HPP_INCLUDED
#include <nt2/include/functor.hpp>


namespace nt2 { namespace tag
  {
   /*!
     @brief ellpe generic tag

     Represents the ellpe function in generic contexts.

     @par Models:
        Hierarchy
   **/
    struct ellpe_ : ext::elementwise_<ellpe_>
    {
      /// @brief Parent hierarchy
      typedef ext::elementwise_<ellpe_> parent;
      template<class... Args>
      static BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE dispatch(Args&&... args)
      BOOST_AUTO_DECLTYPE_BODY( dispatching_ellpe_( ext::adl_helper(), static_cast<Args&&>(args)... ) )
    };
  }
  namespace ext
  {
   template<class Site, class... Ts>
   BOOST_FORCEINLINE generic_dispatcher<tag::ellpe_, Site> dispatching_ellpe_(adl_helper, boost::dispatch::meta::unknown_<Site>, boost::dispatch::meta::unknown_<Ts>...)
   {
     return generic_dispatcher<tag::ellpe_, Site>();
   }
   template<class... Args>
   struct impl_ellpe_;
  }
  /*!
    Complete elliptic integral of the second kind

    @par Semantic:

    For every parameter of floating type T0

    @code
    T0 r = ellpe(a0);
    @endcode

    is similar to:

    @code
    T0 r = ellint_2(a0, Pio_2<T0>());
    @endcode

    @param a0

    @return a value of the same type as the parameter
  **/
  NT2_FUNCTION_IMPLEMENTATION(tag::ellpe_, ellpe, 1)
}

#endif

