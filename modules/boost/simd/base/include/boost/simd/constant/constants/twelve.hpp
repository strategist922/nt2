//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_CONSTANT_CONSTANTS_TWELVE_HPP_INCLUDED
#define BOOST_SIMD_CONSTANT_CONSTANTS_TWELVE_HPP_INCLUDED

#include <boost/simd/include/functor.hpp>
#include <boost/simd/constant/register.hpp>
#include <boost/simd/constant/hierarchy.hpp>


namespace boost { namespace simd
{
  namespace tag
  {
   /*!
     @brief Twelve generic tag

     Represents the Twelve constant in generic contexts.

     @par Models:
        Hierarchy
   **/
    BOOST_SIMD_CONSTANT_REGISTER( Twelve, double, 12
                                ,0x41400000, 0x4028000000000000ll
                                )
  }
  namespace ext
  {
   template<class Site, class... Ts>
   BOOST_FORCEINLINE generic_dispatcher<tag::Twelve, Site> dispatching_Twelve(adl_helper, boost::dispatch::meta::unknown_<Site>, boost::dispatch::meta::unknown_<Ts>...)
   {
     return generic_dispatcher<tag::Twelve, Site>();
   }
   template<class... Args>
   struct impl_Twelve;
  }
  /*!
    Generates value 12

    @par Semantic:

    @code
    T r = Twelve<T>();
    @endcode

    is similar to:

    @code
    T r = T(12);
    @endcode
  **/
  BOOST_SIMD_CONSTANT_IMPLEMENTATION(boost::simd::tag::Twelve, Twelve)
} }

#include <boost/simd/constant/common.hpp>

#endif
