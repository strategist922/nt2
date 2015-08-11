// ==============================================================================
//         Copyright 2013 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI

//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
// ==============================================================================
#ifndef NT2_ARITHMETIC_FUNCTIONS_OPENCL_SQR_HPP_INCLUDED
#define NT2_ARITHMETIC_FUNCTIONS_OPENCL_SQR_HPP_INCLUDED

#include <string>

namespace nt2 { namespace opencl {
inline std::string sqr()
{
  return std::string("{ return exp2(arg0); }");
}
}}

#endif
