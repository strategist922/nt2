//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_OPENMP_UTILITY_HPP_INCLUDED
#define NT2_SDK_OPENMP_UTILITY_HPP_INCLUDED

#if defined(_OPENMP) && _OPENMP >= 200203 /* OpenMP 2.0 */

#include <omp.h>

namespace nt2
{
    inline int get_num_threads()
    {
        return omp_get_max_threads();
    }

    inline void set_num_threads(int n)
    {
        omp_set_num_threads(n);
    }
}

#endif
#endif
