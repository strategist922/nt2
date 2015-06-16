//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef BOOST_SIMD_PREDICATES_FUNCTIONS_SIMD_SSE_SSE_IS_EQZ_HPP_INCLUDED
#define BOOST_SIMD_PREDICATES_FUNCTIONS_SIMD_SSE_SSE_IS_EQZ_HPP_INCLUDED
#ifdef BOOST_SIMD_HAS_SSE_SUPPORT
#include <boost/simd/predicates/functions/is_eqz.hpp>
#include <boost/simd/include/functions/simd/is_equal.hpp>
#include <boost/simd/include/constants/zero.hpp>
#include <boost/simd/include/functions/simd/bitwise_and.hpp>
#include <boost/simd/sdk/meta/as_logical.hpp>
#include <boost/simd/swar/functions/details/shuffle.hpp>
#include <boost/dispatch/meta/downgrade.hpp>

namespace boost { namespace simd { namespace ext
{

} } }

#endif
#endif
