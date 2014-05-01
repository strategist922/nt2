//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_DETAILS_PLASMA_GETRF_INCPIV_HPP_INCLUDED
#define NT2_LINALG_DETAILS_PLASMA_GETRF_INCPIV_HPP_INCLUDED

#include <nt2/linalg/details/utility/plasma_utility.hpp>
#include <nt2/linalg/details/plasma/gessm.hpp>
#include <nt2/linalg/functions/getf2.hpp>

#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>

namespace nt2
{
    template<typename T>
    nt2_la_int getrf_incpiv( nt2_la_int IB,
                             table<T> & A,
                             table< nt2_la_int> IPIV
                            )
    {
        nt2_la_int M = nt2::height(A);
        nt2_la_int N = nt2::width(A);

        nt2_la_int i, j, k, sb;
        nt2_la_int m;

        /* Check input arguments */
        if (M < 0) {
            coreblas_error(1, "Illegal value of M");
            return -1;
        }
        if (N < 0) {
            coreblas_error(2, "Illegal value of N");
            return -2;
        }
        if (IB < 0) {
            coreblas_error(3, "Illegal value of IB");
            return -3;
        }
        if ((LDA < max(1,M)) && (M > 0)) {
            coreblas_error(5, "Illegal value of LDA");
            return -5;
        }

        /* Quick return */
        if ((M == 0) || (N == 0) || (IB == 0))
            return KERNEL_SUCCESS;

        k = min(M, N);

        for(i =0 ; i < k; i += IB) {
            sb = min(IB, k-i);
            /*
             * Factor diagonal and subdiagonal blocks and test for exact singularity.
             */
            m = M-i;
            nt2::getf2(A(_(i+1,i+m),_(i+1,i+sb)), IPIV(_(i+1,i+m)));
            /*
             * Adjust pivot indices.
             */

            if (i+sb < N) {
                nt2::gessm( sb,
                    IPIV(_(i,M)),
                    A(_(i+1,M),_(i+1,i+sb)),
                    A(_(i+1,M),_(i+sb+1,N))
                    );
            }

            for(j = i; j < i+sb; j++) {
                IPIV[j] = i + IPIV[j];
            }
        }
        return 0;
    }
}

#endif
