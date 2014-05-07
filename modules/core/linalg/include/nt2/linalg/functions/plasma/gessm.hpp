//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_PLASMA_GESSM_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_PLASMA_GESSM_HPP_INCLUDED

#include <nt2/linalg/functions/gessm.hpp>
#include <nt2/linalg/functions/laswp.hpp>
#include <nt2/linalg/functions/trsm.hpp>
#include <nt2/linalg/functions/mtimes.hpp>

#include <nt2/linalg/details/utility/plasma_utility.hpp>

#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>

#include <algorithm>

namespace nt2 { namespace ext
{
    NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::gessm_, tag::cpu_
                              , (A0)(A1)(A2)(A3)
                              , (scalar_< integer_<A0> >)
                                ((ast_< A1, nt2::container::domain>))
                                ((ast_< A2, nt2::container::domain>))
                                ((ast_< A3, nt2::container::domain>))
                              )
    {
     typedef nt2_la_int result_type;
     typedef typename A2::value_type T;

     BOOST_FORCEINLINE result_type operator()( A0 const& IB, A1& IPIV, A2& L, A3& A) const
     {
        using nt2::_;

        std::size_t M = nt2::height(A);
        std::size_t N = nt2::width(A);
        std::size_t K = nt2::height(L);

        T mzone = -1.0;

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
        if ((L.leading_size() < std::max(1,M)) && (M > 0)) {
            coreblas_error(7, "Illegal value of LDL");
            return -7;
        }
        if ((A.leading_size() < std::max(1,M)) && (M > 0)) {
            coreblas_error(9, "Illegal value of LDA");
            return -9;
        }

        /* Quick return */
        if ((M == 0) || (N == 0) || (K == 0) || (IB == 0))
            return 0;

        for(std::size_t i = 0; i < K; i += IB) {
            std::size_t sb = std::min(IB, K-i);
            /*
             * Apply nt2_la_interchanges to columns I*IB+1:IB*( I+1 )+1.
             */
            nt2::laswp(boost::proto::value( A(_(1,M),_(1,N))) ),
                       i+1,
                       i+sb,
                       boost::proto::value( IPIV(_(1,M)) )
                       );
            /*
             * Compute block row of U.
             */
            nt2::trsm( 'L', 'L', 'N', 'U',
                     , boost::proto::value( L(_(i+1,i+sb), _(i+1,i+sb)) ),
                     , boost::proto::value( A(_(i+1,i+sb), _(1,N)) )
                     );

            if (i+sb < M) {
            /*
            * Update trailing submatrix.
            */
            A(_(i+sb+1,M),_(1,N)) =
              nt2::mtimes( L(_(i+sb+1,M), _(i+1,i+sb)), A(_(i+1,i+sb),_(1,N)), mzone );
            }
        }
        return 0;
    }
  };

} }

#endif
