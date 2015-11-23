//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_MAGMA_GESVD_W_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_MAGMA_GESVD_W_HPP_INCLUDED

#if defined(NT2_USE_MAGMA)

#include <nt2/linalg/functions/gesvd.hpp>
#include <nt2/include/functions/xerbla.hpp>
#include <nt2/sdk/magma/magma.hpp>
#include <nt2/core/container/table/kind.hpp>

#include <nt2/dsl/functions/terminal.hpp>

#include <nt2/include/functions/of_size.hpp>
#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>
#include <nt2/linalg/details/utility/f77_wrapper.hpp>
#include <nt2/linalg/details/utility/workspace.hpp>

#include <magma.h>
#include <cctype>

namespace nt2 { namespace ext
{
   /// INTERNAL ONLY - Compute the workspace
  BOOST_DISPATCH_IMPLEMENT  ( gesvd_w_, nt2::tag::magma_<site>
                            , (A0)(S0)(A1)(S1)(site)
                            , ((container_<nt2::tag::table_,  double_<A0>, S0 >))
                              ((container_<nt2::tag::table_,  double_<A1>, S1 >))
                            )
  {
     typedef nt2_la_int result_type;

     BOOST_FORCEINLINE result_type operator()( A0& a0, A1& s) const
     {
        result_type that;
        details::workspace<typename A0::value_type> w;
        nt2_la_int  m  = nt2::height(a0);
        nt2_la_int  n  = nt2::width(a0);
        nt2_la_int  ld = a0.leading_size();
        nt2_la_int ldu = 1;
        nt2_la_int ldvt= 1;
        nt2_la_int lwork_query = -1;
        magma_vec_t magma_jobu  = MagmaNoVec ;

        magma_dgesvd(magma_jobu,magma_jobu,m, n, 0, ld, 0, 0, ldu
                            , 0, ldvt, w.main()
                            , lwork_query, &that
                            );

        w.prepare_main();
        nt2::gesvd_w(a0,s,w);

        return that;
     }
  };

  /// INTERNAL ONLY - Workspace is ready
  BOOST_DISPATCH_IMPLEMENT  ( gesvd_w_, nt2::tag::magma_<site>
                            , (A0)(S0)(A1)(S1)(WK)(site)
                            , ((container_<nt2::tag::table_,  double_<A0>, S0 >))
                              ((container_<nt2::tag::table_,  double_<A1>, S1 >))
                              (unspecified_<WK>)

                            )
  {
     typedef nt2_la_int result_type;

     BOOST_FORCEINLINE result_type operator()( A0& a0, A1& s, WK& w) const
     {
        result_type that;
        nt2_la_int  m  = nt2::height(a0);
        nt2_la_int  n  = nt2::width(a0);
        nt2_la_int  ld = a0.leading_size();
        nt2_la_int ldu =  1 ;
        nt2_la_int ldvt=  1 ;
        nt2_la_int  wn = w.main_size();
        magma_vec_t magma_jobu  = MagmaNoVec ;

        magma_dgesvd( magma_jobu,magma_jobu,m, n, a0.data(), ld, s.data(), 0, ldu
                            , 0, ldvt, w.main()
                            , wn, &that
                            );
        return that;
     }
  };


   /// INTERNAL ONLY - Compute the workspace
  BOOST_DISPATCH_IMPLEMENT  ( gesvd_w_, nt2::tag::magma_<site>
                            , (A0)(S0)(A1)(S1)(site)
                            , ((container_<nt2::tag::table_,  single_<A0>, S0 >))
                              ((container_<nt2::tag::table_,  single_<A1>, S1 >))
                            )
  {
     typedef nt2_la_int result_type;

     BOOST_FORCEINLINE result_type operator()( A0& a0, A1& s) const
     {
        result_type that;
        details::workspace<typename A0::value_type> w;
        nt2_la_int  m  = nt2::height(a0);
        nt2_la_int  n  = nt2::width(a0);
        nt2_la_int  ld = a0.leading_size();
        nt2_la_int ldu = 1;
        nt2_la_int ldvt= 1;
        nt2_la_int lwork_query = -1;
        magma_vec_t magma_jobu  = MagmaNoVec ;

        magma_sgesvd(magma_jobu,magma_jobu,m, n, 0, ld, 0, 0, ldu
                            , 0, ldvt, w.main()
                            , lwork_query, &that
                            );

        w.prepare_main();
        nt2::gesvd_w(a0,s,w);

        return that;
     }
  };

  /// INTERNAL ONLY - Workspace is ready
  BOOST_DISPATCH_IMPLEMENT  ( gesvd_w_, nt2::tag::magma_<site>
                            , (A0)(S0)(A1)(S1)(WK)(site)
                            , ((container_<nt2::tag::table_,  single_<A0>, S0 >))
                              ((container_<nt2::tag::table_,  single_<A1>, S1 >))
                              (unspecified_<WK>)
                            )
  {
     typedef nt2_la_int result_type;

     BOOST_FORCEINLINE result_type operator()( A0& a0, A1& s, WK& w) const
     {
        result_type that;
        nt2_la_int  m  = nt2::height(a0);
        nt2_la_int  n  = nt2::width(a0);
        nt2_la_int  ld = a0.leading_size();
        nt2_la_int ldu =  1 ;
        nt2_la_int ldvt=  1 ;
        nt2_la_int  wn = w.main_size();
        magma_vec_t magma_jobu  = MagmaNoVec ;

        magma_sgesvd( magma_jobu,magma_jobu,m, n, a0.data(), ld, s.data(), 0, ldu
                            , 0, ldvt, w.main()
                            , wn, &that
                            );
        return that;
     }
  };

//---------------------------------------------Complex------------------------------------------------//

  /// INTERNAL ONLY - Compute the workspace
  BOOST_DISPATCH_IMPLEMENT  ( gesvd_w_, nt2::tag::magma_<site>
                            , (A0)(S0)(A1)(S1)(site)
                            , ((container_<nt2::tag::table_,  complex_<single_<A0> >, S0 >))
                              ((container_<nt2::tag::table_,  single_<A1>, S1 >))
                            )
  {
     typedef nt2_la_int result_type;

     BOOST_FORCEINLINE result_type operator()( A0& a0, A1& s) const
     {
        result_type that;
        details::workspace<typename A0::value_type> w;
        nt2_la_int  m  = nt2::height(a0);
        nt2_la_int  n  = nt2::width(a0);
        nt2_la_int  ld = a0.leading_size();
        nt2_la_int ldu = 1;
        nt2_la_int ldvt= 1;
        nt2_la_int lwork_query = -1;
        magma_vec_t magma_jobu  = MagmaNoVec ;

        magma_cgesvd(magma_jobu,magma_jobu,m, n, 0, ld, 0, 0, ldu
                            , 0, ldvt, (cuFloatComplex*)w.main()
                            , lwork_query, 0 ,  &that
                            );

        w.prepare_main();
        nt2::gesvd_w(a0,s,w);

        return that;
     }
  };

  /// INTERNAL ONLY - Workspace is ready
  BOOST_DISPATCH_IMPLEMENT  ( gesvd_w_, nt2::tag::magma_<site>
                            , (A0)(S0)(A1)(S1)(WK)(site)
                            , ((container_<nt2::tag::table_,  complex_<single_<A0> >, S0 >))
                              ((container_<nt2::tag::table_,  single_<A1>, S1 >))
                              (unspecified_<WK>)
                            )
  {
     typedef nt2_la_int result_type;

     BOOST_FORCEINLINE result_type operator()( A0& a0, A1& s, WK& w) const
     {
        result_type that;
        nt2_la_int  m  = nt2::height(a0);
        nt2_la_int  n  = nt2::width(a0);
        nt2_la_int  ld = a0.leading_size();
        nt2_la_int ldu =  1 ;
        nt2_la_int ldvt=  1 ;
        nt2_la_int  wn = w.main_size();
        magma_vec_t magma_jobu  = MagmaNoVec ;
        nt2::container::table<float> rwork(nt2::of_size(5*std::min(m,n),1));

        magma_cgesvd( magma_jobu,magma_jobu,m, n, (cuFloatComplex*)a0.data(), ld, s.data()
                            , 0, ldu
                            , 0, ldvt, (cuFloatComplex*)w.main()
                            , wn, rwork.data(), &that
                            );
        return that;
     }
  };

  /// INTERNAL ONLY - Compute the workspace
  BOOST_DISPATCH_IMPLEMENT  ( gesvd_w_, nt2::tag::magma_<site>
                            , (A0)(S0)(A1)(S1)(site)
                            , ((container_<nt2::tag::table_,  complex_<double_<A0> >, S0 >))
                              ((container_<nt2::tag::table_,  double_<A1>, S1 >))
                            )
  {
     typedef nt2_la_int result_type;

     BOOST_FORCEINLINE result_type operator()( A0& a0, A1& s) const
     {
        result_type that;
        details::workspace<typename A0::value_type> w;
        nt2_la_int  m  = nt2::height(a0);
        nt2_la_int  n  = nt2::width(a0);
        nt2_la_int  ld = a0.leading_size();
        nt2_la_int ldu = 1;
        nt2_la_int ldvt= 1;
        nt2_la_int lwork_query = -1;
        magma_vec_t magma_jobu  = MagmaNoVec ;

        magma_zgesvd(magma_jobu,magma_jobu,m, n, 0, ld, 0, 0, ldu
                            , 0, ldvt,(cuDoubleComplex*)w.main()
                            , lwork_query, 0 , &that
                            );

        w.prepare_main();
        nt2::gesvd_w(a0,s,w);

        return that;
     }
  };

  /// INTERNAL ONLY - Workspace is ready
  BOOST_DISPATCH_IMPLEMENT  ( gesvd_w_, nt2::tag::magma_<site>
                            , (A0)(S0)(A1)(S1)(WK)(site)
                            , ((container_<nt2::tag::table_,  complex_<double_<A0> >, S0 >))
                              ((container_<nt2::tag::table_,  double_<A1>, S1 >))
                              (unspecified_<WK>)
                            )
  {
     typedef nt2_la_int result_type;

     BOOST_FORCEINLINE result_type operator()( A0& a0, A1& s, WK& w) const
     {
        result_type that;
        nt2_la_int  m  = nt2::height(a0);
        nt2_la_int  n  = nt2::width(a0);
        nt2_la_int  ld = a0.leading_size();
        nt2_la_int ldu =  1 ;
        nt2_la_int ldvt=  1 ;
        nt2_la_int  wn = w.main_size();
        magma_vec_t magma_jobu  = MagmaNoVec ;
        nt2::container::table<double> rwork(nt2::of_size(5*std::min(m,n),1));

        magma_zgesvd( magma_jobu,magma_jobu,m, n, (cuDoubleComplex*)a0.data(), ld, s.data()
                            , 0, ldu
                            , 0, ldvt, (cuDoubleComplex*)w.main()
                            , wn, rwork.data(), &that
                            );
        return that;
     }
  };
} }

#endif
#endif

