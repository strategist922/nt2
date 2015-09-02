// ==============================================================================
//         Copyright 2013 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI

//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
// ==============================================================================
#ifndef NT2_ARITHMETIC_FUNCTIONS_OPENCL_FASTNORMCDF_HPP_INCLUDED
#define NT2_ARITHMETIC_FUNCTIONS_OPENCL_FASTNORMCDF_HPP_INCLUDED

//#ifdef __OPENCLCC__
//#error(fastnormcdf not defined)

#include <string>

namespace nt2 { namespace opencl {
inline std::string fastnormcdf()
{
  std::string res =
//    "inline float cnd(float arg0)"
    "{"
    "    const float A1 =  0.319381530f;"
    "    const float A2 = -0.356563782f;"
    "    const float A3 =  1.781477937f;"
    "    const float A4 = -1.821255978f;"
    "    const float A5 =  1.330274429f;"
    "    const float RSQRT2PI = 0.39894228040143267793994605993438f;"
    "    float K = 1.0f / (1.0f + 0.2316419f * fabs(arg0));"
    "    float cnd ="
    "        RSQRT2PI * exp(-0.5f * arg0 * arg0) *"
    "        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));"
    "    if(arg0 > 0){"
    "        cnd = 1.0f - cnd;"
    "    }"
    "    return cnd;"
//    "}"
  ;

  return res;
}
}}

//__forceinline__ __device__
//double fastnormcdf(double a)
//{
//  return normcdf(a);
//}
//
//__forceinline__ __device__
//float fastnormcdf(float a)
//{
//  return normcdff(a);
//}


//#endif
#endif
