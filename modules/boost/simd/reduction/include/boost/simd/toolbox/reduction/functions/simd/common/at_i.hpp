//==============================================================================
//         Copyright 2003 - 2011 LASMEA UMR 6602 CNRS/Univ. Clermont II         
//         Copyright 2009 - 2011 LRI    UMR 8623 CNRS/Univ Paris Sud XI         
//                                                                              
//          Distributed under the Boost Software License, Version 1.0.          
//                 See accompanying file LICENSE.txt or copy at                 
//                     http://www.boost.org/LICENSE_1_0.txt                     
//==============================================================================
#ifndef BOOST_SIMD_TOOLBOX_REDUCTION_FUNCTIONS_SIMD_COMMON_AT_I_HPP_INCLUDED
#define BOOST_SIMD_TOOLBOX_REDUCTION_FUNCTIONS_SIMD_COMMON_AT_I_HPP_INCLUDED
#include <boost/simd/sdk/meta/scalar_of.hpp>

namespace boost { namespace simd { namespace ext
{
  BOOST_SIMD_FUNCTOR_IMPLEMENTATION(boost::simd::tag::at_i_, tag::cpu_,
                     (A0)(A1)(X),
                     ((simd_<arithmetic_<A0>,X>))
                     ((scalar_<integer_<A1> >))
                    )
  {
    typedef typename meta::scalar_of<A0>::type result_type;
    BOOST_SIMD_FUNCTOR_CALL(2) { return a0[a1]; }
  };
} } }
#endif
