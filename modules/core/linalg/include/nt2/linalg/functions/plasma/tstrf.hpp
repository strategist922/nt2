//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_DETAILS_PLASMA_TSTRF_HPP_INCLUDED
#define NT2_LINALG_DETAILS_PLASMA_TSTRF_HPP_INCLUDED

#include <nt2/linalg/functions/tstrf.hpp>
#include <nt2/include/functions/ssssm.hpp>
#include <nt2/include/functions/ger.hpp>
#include <nt2/include/functions/iamax.hpp>
#include <nt2/include/functions/swap.hpp>

#include <nt2/linalg/details/utility/plasma_utility.hpp>
#include <nt2/linalg/details/utility/f77_wrapper.hpp>

#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>

#include <nt2/include/functions/evaluate.hpp>

#include <algorithm>

namespace nt2 { namespace ext
{
    NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::tstrf_, tag::cpu_
                              , (A0)(A1)(A2)(A3)(A4)(A5)
                              , (unspecified_<A0>)
                                ((ast_< A1, nt2::container::domain>))
                                ((ast_< A2, nt2::container::domain>))
                                ((ast_< A3, nt2::container::domain>))
                                ((ast_< A4, nt2::container::domain>))
                                ((ast_< A5, nt2::container::domain>))
                              )
    {

     typedef nt2_la_int result_type;
     typedef typename A1::value_type T;

     BOOST_FORCEINLINE result_type operator()(A0 ibnb,
                                              A1 U,
                                              A2 A,
                                              A3 L,
                                              A4 IPIV,
                                              A5 & WORK
                                             ) const
     {
        using nt2::_;

        nt2_la_int IB = ibnb.first;
        nt2_la_int NB = ibnb.second;
        nt2_la_int  M = nt2::height(A);
        nt2_la_int  N = nt2::width(A);

        nt2_la_int  LDU = U.leading_size();
        nt2_la_int  LDA = A.leading_size();
        nt2_la_int  LDL = L.leading_size();

        T zzero = 0.0;
        T mzone =-1.0;
        T alpha;
        nt2_la_int i;

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
        if ((LDU < std::max(1,NB)) && (NB > 0)) {
            coreblas_error(6, "Illegal value of LDU");
            return -6;
        }
        if ((LDA < std::max(1,M)) && (M > 0)) {
            coreblas_error(8, "Illegal value of LDA");
            return -8;
        }
        if ((LDL < std::max(1,IB)) && (IB > 0)) {
            coreblas_error(10, "Illegal value of LDL");
            return -10;
        }

        /* Quick return */
        if ((M == 0) || (N == 0) || (IB == 0))
            return 0;

        /* Set L to 0 */
        L = 0.;

        for (nt2_la_int ii = 0, ip = 0; ii < N; ii += IB, ip++) {

            nt2_la_int sb = std::min(N-ii, IB);

            for (i = 0; i < sb; i++) {
                nt2_la_int im = nt2::iamax(
                                 boost::proto::value(
                                 A(_(1,M), ii+i+1)
                                 ));

                IPIV(ip+1) = ii+i+1;

                if ( std::fabs( A(im+1,ii+i+1) ) > std::fabs( U(ii+i+1,ii+i+1) ) ){
                    /*
                     * Swap behind.
                     */
                    nt2::swap( boost::proto::value( nt2::evaluate(
                               L(i+1, _(ii+1,ii+i))
                               )),
                               boost::proto::value( nt2::evaluate(
                               WORK(im+1,_(1,i))
                               ))
                              );
                    /*
                     * Swap ahead.
                     */
                    nt2::swap( boost::proto::value( nt2::evaluate(
                               U(ii+i+1,_(ii+i+1,ii+sb))
                               )),
                               boost::proto::value( nt2::evaluate(
                               A(im+1,_(ii+i+1,ii+sb))
                               ))
                              );
                    /*
                     * Set IPIV.
                     */
                    IPIV(ip+1) = NB + im + 1;

                    for (nt2_la_int j = 0; j < i; j++) {
                        A(im+1,ii+j+1) = zzero;
                    }
                }

                alpha = ((T)1. / U(ii+i+1,ii+i+1));
                A(_(1,M), ii+i+1) = alpha*A(_(1,M), ii+i+1);
                WORK(_(1,M), i+1) = A(_(1,M), ii+i+1);

                // Warning: first arg must be a column vector, the second must be a line vector
                nt2::ger(   boost::proto::value( nt2::evaluate(
                            A(_(1,M), ii+i+1)
                            ))
                          , boost::proto::value( nt2::evaluate(
                            U(ii+i+1, _(ii+i+2,ii+sb))
                            ))
                          , boost::proto::value( nt2::evaluate(
                            A(_(1,M), _(ii+i+2,ii+sb))
                            ))
                          , mzone
                          );
            }
            /*
             * Apply the subpanel to the rest of the panel.
             */
            if(ii+i < N) {
                for(nt2_la_int j = ii; j < ii+sb; j++) {
                    if (IPIV(j+1) <= NB) {
                        IPIV(j+1) = IPIV(j+1) - ii;
                    }
                }

             nt2::ssssm( sb,
                         nt2::evaluate(
                         U(_(ii+1,ii+NB), _(ii+sb+1,N))
                         ),
                         nt2::evaluate(
                         A(_(1,M), _(ii+sb+1,N))
                         ),
                         nt2::evaluate(
                         L(_(1,sb), _(ii+1,ii+sb))
                         ),
                         nt2::evaluate(
                         WORK(_(1,M),_(1,N-(ii+sb)))
                         ),
                         nt2::evaluate(
                         IPIV(_(ii+1,M))
                         )
                        );

                for(nt2_la_int j = ii; j < ii+sb; j++) {
                    if (IPIV(j+1) <= NB) {
                        IPIV(j+1) = IPIV(j+1) + ii;
                    }
                }
            }
        }
        return 0;
    }
  };

} }

#endif
