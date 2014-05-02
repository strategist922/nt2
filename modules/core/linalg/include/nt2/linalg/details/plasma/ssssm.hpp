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
#include <nt2/linalg/functions/swap.hpp>
#include <nt2/linalg/functions/trsm.hpp>
#include <nt2/linalg/functions/mtimes.hpp>

#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>

#include <algorithm>

namespace nt2
{
    template<typename T>
    nt2_la_int ssssm( nt2_la_int IB,
                nt2::table<T> & A1,
                nt2::table<T> & A2,
                nt2::table<T> & L1,
                nt2::table<T> & L2,
                const nt2::table<nt2_la_int> & IPIV
                )
    {
        nt2_la_int M1 = nt2::height(A1);
        nt2_la_int N1 = nt2::width(A1);
        nt2_la_int M2 = nt2::height(A2);
        nt2_la_int N2 = nt2::width(A2);
        nt2_la_int K  = nt2::height(L1);

        static T mzone =-1.0;

        nt2_la_int i, ii, sb;
        nt2_la_int im, ip;

        /* Check input arguments */
        if (M1 < 0) {
            coreblas_error(1, "Illegal value of M1");
            return -1;
        }
        if (N1 < 0) {
            coreblas_error(2, "Illegal value of N1");
            return -2;
        }
        if (M2 < 0) {
            coreblas_error(3, "Illegal value of M2");
            return -3;
        }
        if (N2 < 0) {
            coreblas_error(4, "Illegal value of N2");
            return -4;
        }
        if (K < 0) {
            coreblas_error(5, "Illegal value of K");
            return -5;
        }
        if (IB < 0) {
            coreblas_error(6, "Illegal value of IB");
            return -6;
        }
        if (A1.leading_size() < std::max(1,M1)) {
            coreblas_error(8, "Illegal value of LDA1");
            return -8;
        }
        if (A2.leading_size() < std::max(1,M2)) {
            coreblas_error(10, "Illegal value of LDA2");
            return -10;
        }
        if (L1.leading_size() < std::max(1,IB)) {
            coreblas_error(12, "Illegal value of LDL1");
            return -12;
        }
        if (L2.leading_size() < std::max(1,M2)) {
            coreblas_error(14, "Illegal value of LDL2");
            return -14;
        }

        /* Quick return */
        if ((M1 == 0) || (N1 == 0) || (M2 == 0) || (N2 == 0) || (K == 0) || (IB == 0))
            return KERNEL_SUCCESS;

        ip = 0;

        for(ii = 0; ii < K; ii += IB) {
            sb = std::min(K-ii, IB);

            for(i = 0; i < sb; i++) {
                im = IPIV[ip]-1;

                if (im != (ii+i)) {
                    im = im - M1;
                    nt2::swap(A1(_(ii+i+1,ii+i+N1),_), A2(_(im+1,im+N1),_));
                }
                ip = ip + 1;
            }

            nt2::trsm(
                'L', 'L', 'N', 'U',
                L1(_(1,sb),_(ii+1,ii+sb)),
                A1(_(ii+1,ii+sb),_)
                );

            A2 = nt2::mtimes(
                'N','N'
                L2(_,_(ii+1,ii+sb)),
                A1(_(ii+1,ii+sb),_(1,N2)),
                mzone
                );
        }
        return KERNEL_SUCCESS;
    }
}

#endif
