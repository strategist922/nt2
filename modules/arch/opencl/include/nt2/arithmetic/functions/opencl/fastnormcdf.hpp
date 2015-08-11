// ==============================================================================
//         Copyright 2013 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI

//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
// ==============================================================================
#ifndef NT2_ARITHMETIC_FUNCTIONS_OPENCL_FASTNORMCDF_HPP_INCLUDED
#define NT2_ARITHMETIC_FUNCTIONS_OPENCL_FASTNORMCDF_HPP_INCLUDED

#ifdef __OPENCLCC__
#error(fastnormcdf not defined)

__forceinline__ __device__
double fastnormcdf(double a)
{
  return normcdf(a);
}

__forceinline__ __device__
float fastnormcdf(float a)
{
  return normcdff(a);
}


#endif
#endif
