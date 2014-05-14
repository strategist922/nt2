//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_PLASMA_GETRF_INCPIV_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_PLASMA_GETRF_INCPIV_HPP_INCLUDED

#include <nt2/linalg/functions/getrf_incpiv.hpp>
#include <nt2/include/functions/gessm.hpp>
#include <nt2/include/functions/getf2.hpp>

#include <nt2/linalg/details/utility/plasma_utility.hpp>

#include <nt2/include/functions/evaluate.hpp>
#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>

#include <algorithm>

namespace nt2 { namespace ext
{
    NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::getrf_incpiv_, tag::cpu_
                              , (A0)(A1)(A2)
                              , (scalar_< integer_<A0> >)
                                ((ast_< A1, nt2::container::domain>))
                                ((ast_< A2, nt2::container::domain>))
                              )
    {
     typedef int result_type;
     typedef typename A1::value_type T;

     BOOST_FORCEINLINE result_type operator()( A0 const & IB, A1 A, A2 IPIV) const
     {
        using nt2::_;

        nt2_la_int M = nt2::height(A);
        nt2_la_int N = nt2::width(A);
        nt2_la_int LDA = A.leading_size();

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
        if ((LDA < std::max(1,M)) && (M > 0)) {
            coreblas_error(5, "Illegal value of LDA");
            return -5;
        }

        /* Quick return */
        if ((M == 0) || (N == 0) || (IB == 0))
            return 0;

        nt2_la_int k = std::min(M, N);

        for(nt2_la_int i =0 ; i < k; i += IB) {

            nt2_la_int sb = std::min(IB, k-i);
            /*
             * Factor diagonal and subdiagonal blocks and test for exact singularity.
             */
            nt2::getf2( boost::proto::value( nt2::evaluate( A(_(i+1,M),_(i+1,i+sb)) ) ),
                        boost::proto::value( nt2::evaluate( IPIV(_(i+1,M)) ) )
                      );
            /*
             * Adjust pivot indices.
             */

            if (i+sb < N) {
                nt2::gessm( sb,
                    nt2::evaluate( IPIV(_(i+1,M)) ),
                    nt2::evaluate( A(_(i+1,M),_(i+1,i+sb)) ),
                    nt2::evaluate( A(_(i+1,M),_(i+sb+1,N)) )
                    );
            }

            for(nt2_la_int j = i; j < i+sb; j++) {
                IPIV(j) = i + IPIV(j);
          }
        }
        return 0;
    }
  };

} }

#endif
