//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_DETAILS_PLASMA_GESSM_HPP_INCLUDED
#define NT2_LINALG_DETAILS_PLASMA_GESSM_HPP_INCLUDED

#include <nt2/linalg/details/utility/plasma_utility.hpp>
#include <nt2/linalg/functions/laswp.hpp>
#include <nt2/linalg/functions/trsm.hpp>
#include <nt2/linalg/functions/mtimes.hpp>

#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>

#include <algorithm>

namespace nt2
{
    template<typename T>
    nt2_la_int gessm(nt2_la_int IB,
                     const table<nt2_la_int> & IPIV,
                     const table<T> & L,
                     table<T> & A)
    {
        nt2_la_int M = nt2::height(A);
        nt2_la_int N = nt2::width(A);
        nt2_la_int K = nt2::height(L);

        static T mzone = -1.0;
        static nt2_la_int ione  =  1;

        nt2_la_int i, sb;
        nt2_la_int tmp, tmp2;

        /* Check input arguments */
        if (M < 0) {
            coreblas_error(1, "Illegal value of M");
            return -1;
        }
        if (N < 0) {
            coreblas_error(2, "Illegal value of N");
            return -2;
        }
        if (K < 0) {
            coreblas_error(3, "Illegal value of K");
            return -3;
        }
        if (IB < 0) {
            coreblas_error(4, "Illegal value of IB");
            return -4;
        }
        if ((L.leading_size() < max(1,M)) && (M > 0)) {
            coreblas_error(7, "Illegal value of LDL");
            return -7;
        }
        if ((A.leading_size() < max(1,M)) && (M > 0)) {
            coreblas_error(9, "Illegal value of LDA");
            return -9;
        }

        /* Quick return */
        if ((M == 0) || (N == 0) || (K == 0) || (IB == 0))
            return 0;

        for(i = 0; i < K; i += IB) {
            sb = std::min(IB, K-i);
            /*
             * Apply nt2_la_interchanges to columns I*IB+1:IB*( I+1 )+1.
             */
            nt2::laswp(A(_(1,M),_(1,N)), i+1, i+sb, IPIV(_(1,M));
            /*
             * Compute block row of U.
             */
            nt2::trsm( 'L', 'L', 'N', 'U',
                     , L(_(i+1,i+sb), _(i+1,i+sb)),
                     , A(_(i+1,i+sb), _)
                     );

            if (i+sb < M) {
            /*
            * Update trailing submatrix.
            */
            A(_(i+sb+1,M),_) = nt2::mtimes( L(_(i+sb+1,M), _(i+1,i+sb)), A(_(i+1,i+sb),_), mzone )
            }
        }
        return 0;
    }
}

#endif
