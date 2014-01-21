//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SIMD_CATEGORY_HPP_INCLUDED
#define NT2_SDK_SIMD_CATEGORY_HPP_INCLUDED

#include <boost/simd/sdk/simd/category.hpp>
#include <boost/simd/sdk/simd/logical.hpp>
#include <boost/simd/constant/hierarchy.hpp>

namespace nt2 { namespace BOOST_SIMD_EXT_NS
{
  using boost::simd::BOOST_SIMD_EXT_NS::simd_;
  using boost::simd::BOOST_SIMD_EXT_NS::logical_;

  using boost::simd::BOOST_SIMD_EXT_NS::elementwise_;
  using boost::simd::BOOST_SIMD_EXT_NS::reduction_;
  using boost::simd::BOOST_SIMD_EXT_NS::cumulative_;
  using boost::simd::BOOST_SIMD_EXT_NS::constant_;
} }

#endif
