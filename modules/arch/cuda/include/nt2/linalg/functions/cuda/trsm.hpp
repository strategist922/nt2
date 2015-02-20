//==============================================================================
//         Copyright 2003 - 2012   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_LINALG_FUNCTIONS_CUDA_TRSM_HPP_INCLUDED
#define NT2_LINALG_FUNCTIONS_CUDA_TRSM_HPP_INCLUDED

#if defined(NT2_HAS_CUDA)

#include <nt2/linalg/functions/trsm.hpp>
#include <nt2/sdk/cuda/cuda.hpp>

#include <nt2/dsl/functions/terminal.hpp>
#include <nt2/core/container/table/kind.hpp>
#include <nt2/sdk/meta/device.hpp>

#include <nt2/linalg/details/utility/f77_wrapper.hpp>
#include <nt2/linalg/details/blas/blas3.hpp>
#include <nt2/include/functions/of_size.hpp>
#include <nt2/include/functions/height.hpp>
#include <nt2/include/functions/width.hpp>
#include <complex>
#include <cublas.h>

namespace nt2 { namespace ext
{

  BOOST_DISPATCH_IMPLEMENT ( trsm_, nt2::tag::cuda_<site>
                            , (A0)(A1)(A2)(A3)(A4)(S4)(A5)(S5)(site)
                            , (scalar_< ints8_<A0> >)
                              (scalar_< ints8_<A1> >)
                              (scalar_< ints8_<A2> >)
                              (scalar_< ints8_<A3> >)
                              ((container_< nt2::tag::table_, double_<A4>, S4 >))
                              ((container_< nt2::tag::table_, double_<A5>, S5 >))
                            )
  {
    typedef void  result_type;

     BOOST_FORCEINLINE result_type operator()( A0 const& side, A1 const& uplo, A2 const& transa
                                             , A3 const& diag, A4 const& a, A5& b ) const
     {
        int  m  = nt2::height(a);
        int  n  = nt2::width(b);
        int  lda = (side=='L'||'l') ? m : n ;
        double alpha = 1.;

        auto device_a = to_device(a);
        auto device_b = to_device(b);

        cublasDtrsm( side, uplo, transa, diag, m, n, alpha
                   , device_a.data(), lda, device_b.data(), m);

        device_swap(device_b,b);
     }
  };

  BOOST_DISPATCH_IMPLEMENT( trsm_,  nt2::tag::cuda_<site>
                            , (A0)(A1)(A2)(A3)(A4)(S4)(A5)(S5)(A6)(site)
                            , (scalar_< ints8_<A0> >)
                              (scalar_< ints8_<A1> >)
                              (scalar_< ints8_<A2> >)
                              (scalar_< ints8_<A3> >)
                              ((container_< nt2::tag::table_, double_<A4>, S4 >))
                              ((container_< nt2::tag::table_, double_<A5>, S5 >))
                              (scalar_< floating_<A6> >)
                            )
  {
    typedef void  result_type;

     BOOST_FORCEINLINE result_type operator()( A0 const& side, A1 const& uplo, A2 const& transa
                                             , A3 const& diag, A4 const& a, A5& b, A6 const alpha
                                              ) const
     {
        int  m  = nt2::height(a);
        int  n  = nt2::width(b);
        int  lda = (side=='L'||'l') ? m : n ;

        auto device_a = to_device(a);
        auto device_b = to_device(b);

        cublasDtrsm( side, uplo, transa, diag, m, n, alpha
                   , device_a.data(), lda, (double*)device_b.data(), m);

        device_swap(device_b,b);
     }
  };

  BOOST_DISPATCH_IMPLEMENT( trsm_,  nt2::tag::cuda_<site>
                            , (A0)(A1)(A2)(A3)(A4)(S4)(A5)(S5)(site)
                            , (scalar_< ints8_<A0> >)
                              (scalar_< ints8_<A1> >)
                              (scalar_< ints8_<A2> >)
                              (scalar_< ints8_<A3> >)
                              ((container_< nt2::tag::table_, single_<A4>, S4 >))
                              ((container_< nt2::tag::table_, single_<A5>, S5 >))
                            )
  {
    typedef void  result_type;

     BOOST_FORCEINLINE result_type operator()( A0 const& side, A1 const& uplo, A2 const& transa
                                             , A3 const& diag, A4 const& a, A5& b ) const
     {
        int  m  = nt2::height(a);
        int  n  = nt2::width(b);
        int  lda = (side=='L'||'l') ? m : n ;

        double alpha = 1.;

        auto device_a = to_device(a);
        auto device_b = to_device(b);

        cublasStrsm( side, uplo, transa, diag, m, n, alpha
                   , device_a.data(),lda,device_b.data(),m);

        device_swap(device_b,b);
     }
  };

  BOOST_DISPATCH_IMPLEMENT( trsm_,  nt2::tag::cuda_<site>
                            , (A0)(A1)(A2)(A3)(A4)(S4)(A5)(S5)(site)
                            , (scalar_< ints8_<A0> >)
                              (scalar_< ints8_<A1> >)
                              (scalar_< ints8_<A2> >)
                              (scalar_< ints8_<A3> >)
                              ((container_< nt2::tag::table_, complex_<double_<A4> >, S4 >))
                              ((container_< nt2::tag::table_, complex_<double_<A5> >, S5 >))
                            )
  {
    typedef void  result_type;

     BOOST_FORCEINLINE result_type operator()( A0 const& side, A1 const& uplo, A2 const& transa
                                             , A3 const& diag, A4 const& a, A5& b ) const
     {
        int  m  = nt2::height(a);
        int  n  = nt2::width(b);
        int  lda = (side=='L'||'l') ? m : n ;

        cuDoubleComplex alpha {1. , 0};

        auto device_a = to_device(a);
        auto device_b = to_device(b);

        cublasZtrsm( side, uplo, transa, diag, m, n, alpha
                          , (cuDoubleComplex*)device_a.data(),lda
                          , (cuDoubleComplex*)device_b.data(),m);

        device_swap(device_b,b);
     }
  };

  BOOST_DISPATCH_IMPLEMENT( trsm_,  nt2::tag::cuda_<site>
                            , (A0)(A1)(A2)(A3)(A4)(S4)(A5)(S5)(site)
                            , (scalar_< ints8_<A0> >)
                              (scalar_< ints8_<A1> >)
                              (scalar_< ints8_<A2> >)
                              (scalar_< ints8_<A3> >)
                              ((container_< nt2::tag::table_, complex_<single_<A4> >, S4 >))
                              ((container_< nt2::tag::table_, complex_<single_<A5> >, S5 >))
                            )
  {
    typedef void  result_type;

     BOOST_FORCEINLINE result_type operator()( A0 const& side, A1 const& uplo, A2 const& transa
                                             , A3 const& diag, A4 const& a, A5& b ) const
     {
        nt2_la_int  m  = nt2::height(a);
        nt2_la_int  n  = nt2::width(b);
        nt2_la_int  lda = (side=='L'||'l')? m : n ;

        auto device_a = to_device(a);
        auto device_b = to_device(b);

        cuFloatComplex alphac {1.0,0};

        cublasCtrsm( side, uplo, transa, diag, m, n, alphac
                          , (cuFloatComplex*)device_a.data(),lda
                          , (cuFloatComplex*)device_b.data(), m);

        device_swap(device_b,b);
     }
  };

  BOOST_DISPATCH_IMPLEMENT( trsm_,  nt2::tag::cuda_<site>
                            , (A0)(A1)(A2)(A3)(A4)(S4)(A5)(S5)(A6)(site)
                            , (scalar_< ints8_<A0> >)
                              (scalar_< ints8_<A1> >)
                              (scalar_< ints8_<A2> >)
                              (scalar_< ints8_<A3> >)
                              ((container_< nt2::tag::table_, single_<A4>, S4 >))
                              ((container_< nt2::tag::table_, single_<A5>, S5 >))
                              (scalar_< floating_<A6> >)
                            )
  {
    typedef void  result_type;

     BOOST_FORCEINLINE result_type operator()( A0 const& side, A1 const& uplo, A2 const& transa
                                             , A3 const& diag, A4 const& a, A5& b, A6 const alpha ) const
     {
        int  m  = nt2::height(a);
        int  n  = nt2::width(b);
        int  lda = (side=='L'||'l') ? m : n ;

        auto device_a = to_device(a);
        auto device_b = to_device(b);

        cublasStrsm( side, uplo, transa, diag, m, n, alpha
                          , device_a.data(),lda,(float*)device_b.data(),m);

        device_swap(device_b,b);
     }
  };

  BOOST_DISPATCH_IMPLEMENT( trsm_,  nt2::tag::cuda_<site>
                            , (A0)(A1)(A2)(A3)(A4)(S4)(A5)(S5)(A6)(site)
                            , (scalar_< ints8_<A0> >)
                              (scalar_< ints8_<A1> >)
                              (scalar_< ints8_<A2> >)
                              (scalar_< ints8_<A3> >)
                              ((container_< nt2::tag::table_, complex_<double_<A4> >, S4 >))
                              ((container_< nt2::tag::table_, complex_<double_<A5> >, S5 >))
                              (scalar_< complex_<floating_<A6> > >)
                            )
  {
    typedef void  result_type;

     BOOST_FORCEINLINE result_type operator()( A0 const& side, A1 const& uplo, A2 const& transa
                                             , A3 const& diag, A4 const& a, A5& b, A6 const alpha
                                              ) const
     {
        int  m  = nt2::height(a);
        int  n  = nt2::width(b);
        int  lda = (side=='L'||'l') ? m : n ;

        auto device_a = to_device(a);
        auto device_b = to_device(b);

        cuDoubleComplex alphac {alpha.real(), alpha.imag()};

        cublasZtrsm( side, uplo, transa, diag, m, n, alphac
                          , (cuDoubleComplex*)device_a.data(),lda
                          , (cuDoubleComplex*)device_b.data(),m);

        device_swap(device_b,b);

     }
  };

  BOOST_DISPATCH_IMPLEMENT( trsm_,  nt2::tag::cuda_<site>
                            , (A0)(A1)(A2)(A3)(A4)(S4)(A5)(S5)(A6)(site)
                            , (scalar_< ints8_<A0> >)
                              (scalar_< ints8_<A1> >)
                              (scalar_< ints8_<A2> >)
                              (scalar_< ints8_<A3> >)
                              ((container_< nt2::tag::table_, complex_<single_<A4> >, S4 >))
                              ((container_< nt2::tag::table_, complex_<single_<A5> >, S5 >))
                              (scalar_< complex_<floating_<A6> > >)
                            )
  {
    typedef void  result_type;

     BOOST_FORCEINLINE result_type operator()( A0 const& side, A1 const& uplo, A2 const& transa
                                             , A3 const& diag, A4 const& a, A5& b ,A6 const alpha
                                             ) const
     {
        int  m  = nt2::height(a);
        int  n  = nt2::width(b);
        int  lda = (side=='L'||'l') ? m : n ;

        auto device_a = to_device(a);
        auto device_b = to_device(b);

        cuFloatComplex alphac {alpha.real(), alpha.imag()};

        cublasCtrsm( side, uplo, transa, diag, m, n, alphac
                          , (cuFloatComplex*)device_a.data(),lda
                          , (cuFloatComplex*)device_b.data(),m);

        device_swap(device_b,b);
     }
  };

} }

#endif

#endif
