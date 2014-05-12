//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_PLASMA_SSSSM_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_PLASMA_SSSSM_HPP_INCLUDED

#include <nt2/linalg/functions/ssssm.hpp>
#include <nt2/include/functions/swap.hpp>
#include <nt2/include/functions/trsm.hpp>
#include <nt2/include/functions/mtimes.hpp>

#include <nt2/linalg/details/utility/plasma_utility.hpp>

#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>

#include <algorithm>

namespace nt2 { namespace ext
{
    NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::ssssm_, tag::cpu_
                              , (A0)(A1)(A2)(A3)(A4)(A5)
                              , (scalar_< integer_<A0> >)
                                ((ast_< A1, nt2::container::domain>))
                                ((ast_< A2, nt2::container::domain>))
                                ((ast_< A3, nt2::container::domain>))
                                ((ast_< A4, nt2::container::domain>))
                                ((ast_< A5, nt2::container::domain>))
                              )
    {

     typedef int result_type;
     typedef typename A1::value_type T;

     BOOST_FORCEINLINE result_type operator()( A0 const & IB,
                                               A1 a1,
                                               A2 a2,
                                               A3 L1,
                                               A4 L2,
                                               A5 IPIV
                                             ) const
     {
        using nt2::_;

        nt2_la_int M1 = nt2::height(a1);
        nt2_la_int N1 = nt2::width(a1);
        nt2_la_int M2 = nt2::height(a2);
        nt2_la_int N2 = nt2::width(a2);
        nt2_la_int K  = nt2::height(L1);

        nt2_la_int LDA1 = a1.leading_size();
        nt2_la_int LDA2 = a2.leading_size();
        nt2_la_int LDL1 = L1.leading_size();
        nt2_la_int LDL2 = L2.leading_size();

        static T mzone =-1.0;

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
        if ( LDA1 < std::max(1,M1)) {
            coreblas_error(8, "Illegal value of LDA1");
            return -8;
        }
        if ( LDA2 < std::max(1,M2)) {
            coreblas_error(10, "Illegal value of LDA2");
            return -10;
        }
        if ( LDL1 < std::max(1,IB)) {
            coreblas_error(12, "Illegal value of LDL1");
            return -12;
        }
        if ( LDL2 < std::max(1,M2)) {
            coreblas_error(14, "Illegal value of LDL2");
            return -14;
        }

        /* Quick return */
        if ((M1 == 0) || (N1 == 0) || (M2 == 0) || (N2 == 0) || (K == 0) || (IB == 0))
            return 0;

        nt2_la_int ip = 0;

        for(nt2_la_int ii = 0; ii < K; ii += IB) {
            nt2_la_int sb = std::min(K-ii, IB);

            for(nt2_la_int i = 0; i < sb; i++) {
                nt2_la_int im = IPIV(ip)-1;

                if (im != (ii+i)) {
                    im = im - M1;
                    nt2::swap(
                        boost::proto::value( nt2::evaluate(
                        a1(ii+i+1,_(1,N1))
                        )),
                        boost::proto::value( nt2::evaluate(
                        a2(im+1,_(1,N1))
                        ))
                        );
                }
                ip = ip + 1;
            }

            nt2::trsm(
                'L', 'L', 'N', 'U',
                boost::proto::value( nt2::evaluate(
                L1(_(1,sb),_(ii+1,ii+sb))
                )),
                boost::proto::value( nt2::evaluate(
                a1(_(ii+1,ii+sb),_(1,N1))
                ))
                );

            a2 = nt2::mtimes(
                nt2::evaluate(
                L2(_(1,M2),_(ii+1,ii+sb))
                ),
                nt2::evaluate(
                a1(_(ii+1,ii+sb),_(1,N2))
                ),
                mzone
                );
        }
        return 0;
    }
  };

} }

#endif
