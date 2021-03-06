//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_BITWISE_FUNCTIONS_BITWISE_NOTAND_HPP_INCLUDED
#define BOOST_SIMD_BITWISE_FUNCTIONS_BITWISE_NOTAND_HPP_INCLUDED
#include <boost/simd/include/functor.hpp>
#include <boost/dispatch/include/functor.hpp>
namespace boost { namespace simd { namespace tag
  {
    /*!
      @brief  bitwise_notand generic tag

      Represents the bitwise_notand function in generic contexts.

      @par Models:
      Hierarchy
    **/
    struct bitwise_notand_ : ext::elementwise_<bitwise_notand_>
    {
      /// @brief Parent hierarchy
      typedef ext::elementwise_<bitwise_notand_> parent;
      template<class... Args>
      static BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE dispatch(Args&&... args)
      BOOST_AUTO_DECLTYPE_BODY( dispatching_bitwise_notand_( ext::adl_helper(), static_cast<Args&&>(args)... ) )
    };
  }
  namespace ext
  {
    template<class Site, class... Ts>
    BOOST_FORCEINLINE generic_dispatcher<tag::bitwise_notand_, Site> dispatching_bitwise_notand_(adl_helper, boost::dispatch::meta::unknown_<Site>, boost::dispatch::meta::unknown_<Ts>...)
    {
      return generic_dispatcher<tag::bitwise_notand_, Site>();
    }
    template<class... Args>
    struct impl_bitwise_notand_;
  }
  /*!
    Computes the bitwise and not of its parameters.

    @par semantic:
    For any given value @c x, of type @c T1 @c y of type @c T2
    of same memory size:

    @code
    T1 r = bitwise_notand(x, y);
    @endcode

    The code is equivalent to:

    @code
    T1 r = ~x & y;
    @endcode

    @par Alias

    b_notand

    @see  @funcref{bitwise_or}, @funcref{bitwise_xor}, @funcref{bitwise_and},
    @funcref{bitwise_andnot}, @funcref{bitwise_notor}, @funcref{bitwise_ornot}, @funcref{complement}

    @param  a0
    @param  a1

    @return      a value of the same type as the first input.

  **/
  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION(tag::bitwise_notand_, bitwise_notand, 2)
  BOOST_DISPATCH_FUNCTION_IMPLEMENTATION(tag::bitwise_notand_, b_notand, 2)
} }

#endif

// modified by jt the 25/12/2010
