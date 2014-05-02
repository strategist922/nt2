//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_DETAILS_PLASMA_SSSSM_HPP_INCLUDED
#define NT2_LINALG_DETAILS_PLASMA_SSSSM_HPP_INCLUDED

#include <nt2/linalg/details/utility/plasma_utility.hpp>
#include <nt2/linalg/functions/ger.hpp>
#include <nt2/linalg/functions/iamax.hpp>
#include <nt2/linalg/functions/swap.hpp>
#include <nt2/linalg/details/plasma/ssssm.hpp>

#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>

#include <algorithm>

namespace nt2
{
    template<typename T>
    nt2_la_int tstrf(nt2_la_int IB, nt2_la_int NB,
                     table<T> & U,
                     table<T> & A,
                     table<T> & L,
                     table<nt2_la_int> & IPIV,
                     table<T> & WORK
                     )
    {

        nt2_la_int M = nt2::height(A);
        nt2_la_int N = nt2::width(A);

        static T zzero = 0.0;
        static T mzone =-1.0;

        T alpha;
        nt2_la_int i, j, ii, sb;
        nt2_la_int im, ip;

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
        if ((U.leading_size() < std::max(1,NB)) && (NB > 0)) {
            coreblas_error(6, "Illegal value of LDU");
            return -6;
        }
        if ((A.leading_size() < std::max(1,M)) && (M > 0)) {
            coreblas_error(8, "Illegal value of LDA");
            return -8;
        }
        if ((L.leading_size() < std::max(1,IB)) && (IB > 0)) {
            coreblas_error(10, "Illegal value of LDL");
            return -10;
        }

        /* Quick return */
        if ((M == 0) || (N == 0) || (IB == 0))
            return 0;

        /* Set L to 0 */
        L = 0.;

        ip = 0;
        for (ii = 0; ii < N; ii += IB) {
            sb = min(N-ii, IB);

            for (i = 0; i < sb; i++) {
                im = nt2::iamax(_(1,M), ii+i+1);
                IPIV(ip+1) = ii+i+1;

                if (fabs( A(im+1,ii+i+1) ) > fabs( U(ii+i+1,ii+i+1) ) {
                    /*
                     * Swap behind.
                     */
                    nt2::swap( L(_(1,i), ii+i+1), WORK(_(im+1,im+i)) );
                    /*
                     * Swap ahead.
                     */
                    nt2::swap(U(_(ii+i+1,ii+sb), ii+i+1), A(_(im+1,im+sb-i), ii+i+1) );
                    /*
                     * Set IPIV.
                     */
                    IPIV(ip+1) = NB + im + 1;

                    for (j = 0; j < i; j++) {
                        A(im+1,ii+j+1) = zzero;
                    }
                }

                alpha = ((T)1. / U(ii+i+1,ii+i+1));
                A(_(1,M), ii+i+1) = alpha*A(_(1,M), ii+i+1);
                WORK(_(1,M), i+1) = A(_(1,M), ii+i+1);

                // Warning: first arg must be a column vector, the second must be a line vector
                nt2::ger(   A(_(1,M), ii+i+1)
                          , U(ii+i+1, _(ii+i+2,ii+sb))
                          , A(_(1,M), _(ii+i+2,ii+sb))
                          );
                ip = ip+1;
            }
            /*
             * Apply the subpanel to the rest of the panel.
             */
            if(ii+i < N) {
                for(j = ii; j < ii+sb; j++) {
                    if (IPIV(j+1) <= NB) {
                        IPIV(j+1) = IPIV(j+1) - ii;
                    }
                }

            nt2_la_int nt2::ssssm( sb,
                             U(_(ii+1,ii+NB), _(ii+sb+1,N)),
                             A(_(1,M), _(ii+sb+1,N)),
                             L(_(1,sb), _(ii+1,ii+sb)),
                             WORK(_(1,M),_(1,N-(ii+sb))),
                             IPIV(_(ii+1,M))
                            );

                for(j = ii; j < ii+sb; j++) {
                    if (IPIV(j+1) <= NB) {
                        IPIV(j+1) = IPIV(j+1) + ii;
                    }
                }
            }
        }
        return 0;
    }
}

#endif
