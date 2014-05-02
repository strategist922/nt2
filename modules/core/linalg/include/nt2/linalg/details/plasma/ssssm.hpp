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

namespace nt2 { namespace ext
{
    NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::ssssm_, tag::cpu_
                              , (A0)(A1)(S1)(A2)(S2)(A3)(S3)(A4)(S4)(A5)(S5)
                              , (scalar_< integer_<A0> >)
                                ((container_< nt2::tag::table_, unspecified_<A1>, S1 >))
                                ((container_< nt2::tag::table_, unspecified_<A2>, S2 >))
                                ((container_< nt2::tag::table_, unspecified_<A3>, S3 >))
                                ((container_< nt2::tag::table_, unspecified_<A4>, S4 >))
                                ((container_< nt2::tag::table_, integer_<A5>, S5 >))
                              )
    {

     typedef nt2_la_int result_type;
     typedef typename A1::value_type T;

     BOOST_FORCEINLINE result_type operator()( A0 const & IB,
                                               A1 & a1,
                                               A2 & a2,
                                               A3 & L1,
                                               A4 & L2,
                                               A5 & IPIV
                                             ) const
     {
        std::size_t M1 = nt2::height(a1);
        std::size_t N1 = nt2::width(a1);
        std::size_t M2 = nt2::height(a2);
        std::size_t N2 = nt2::width(a2);
        std::size_t K  = nt2::height(L1);

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
        if (a1.leading_size() < std::max(1,M1)) {
            coreblas_error(8, "Illegal value of LDA1");
            return -8;
        }
        if (a2.leading_size() < std::max(1,M2)) {
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
            return 0;

        for(std::size_t ii = 0, ip =0; ii < K; ii += IB, ip++) {
            std::size_t sb = std::min(K-ii, IB);

            for(i = 0; i < sb; i++) {
                im = IPIV[ip]-1;

                if (im != (ii+i)) {
                    im = im - M1;
                    nt2::swap( a1(ii+i+1,_(1,N1)), a2(im+1,_(1,N1)) );
                }
                ip = ip + 1;
            }

            nt2::trsm(
                'L', 'L', 'N', 'U',
                L1(_(1,sb),_(ii+1,ii+sb)),
                a1(_(ii+1,ii+sb),_)
                );

            a2 = nt2::mtimes(
                'N','N'
                L2(_,_(ii+1,ii+sb)),
                a1(_(ii+1,ii+sb),_(1,N2)),
                mzone
                );
        }
        return 0;
    }
  };

} }

#endif
