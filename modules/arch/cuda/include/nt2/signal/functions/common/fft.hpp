//==============================================================================
//         Copyright 2015 NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SIGNAL_FUNCTIONS_COMMON_FFT_HPP_INCLUDED
#define NT2_SIGNAL_FUNCTIONS_COMMON_FFT_HPP_INCLUDED

#if defined(NT2_HAS_CUDA)

#include <nt2/sdk/cuda/cuda.hpp>
#include <nt2/signal/functions/fft.hpp>
#include <nt2/include/functions/numel.hpp>
#include <nt2/core/container/dsl/as_terminal.hpp>
#include <nt2/core/utility/assign_swap.hpp>
#include <cufft.h>
#include <cuda_runtime.h>

namespace nt2 { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT  ( fft_, nt2::tag::cuda_<site>
                            , (A0)(S0)(A1)(S1)(site)
                            , ((container_< nt2::tag::table_, single_<A0>, S0 >))
                              ((container_< nt2::tag::table_, complex_<single_<A1>>, S1 >))
                            )
  {
    typedef void result_type;
    typedef typename A0::value_type s_type;

    BOOST_FORCEINLINE result_type operator()(A0 const& in, A1& out) const
    {
      out.resize(nt2::extent(in));

      cufftHandle plan;

      CUDA_ERROR(cufftPlan1d(&plan , nt2::numel(in) , CUFFT_R2C , 1));

      CUDA_ERROR(cufftExecR2C( plan
                  ,  in.data()
                  , (cufftComplex *) out.data()
                  , CUFFT_FORWARD
                  ));

      CUDA_ERROR(cufftDestroy(p));
    }
  };

  BOOST_DISPATCH_IMPLEMENT  ( fft_, nt2::tag::cuda_<site>
                            , (A0)(S0)(A1)(S1)(site)
                            , ((container_< nt2::tag::table_, complex_<single_<A0>>, S0 >))
                              ((container_< nt2::tag::table_, complex_<single_<A1>>, S1 >))
                            )
  {
    typedef void result_type;
    typedef typename A0::value_type s_type;

    BOOST_FORCEINLINE result_type operator()(A0 const& in, A1& out) const
    {
      out.resize(nt2::extent(in));

      cufftHandle plan;

      CUDA_ERROR(cufftPlan1d(&plan , nt2::numel(in) , CUFFT_C2C , 1));

      CUDA_ERROR(cufftExecC2C( plan
                  , (cufftComplex *) in.data()
                  , (cufftComplex *) out.data()
                  , CUFFT_FORWARD
                  ));

      CUDA_ERROR(cufftDestroy(p));
    }
  };

  BOOST_DISPATCH_IMPLEMENT  ( fft_, nt2::tag::cuda_<site>
                            , (A0)(S0)(A1)(S1)(site)
                            , ((container_< nt2::tag::table_, double_<A0>, S0 >))
                              ((container_< nt2::tag::table_, complex_<double_<A1>>, S1 >))
                            )
  {
    typedef void result_type;
    typedef typename A0::value_type s_type;

    BOOST_FORCEINLINE result_type operator()(A0 const& in, A1& out) const
    {
      out.resize(nt2::extent(in));

      cufftHandle plan;

      CUDA_ERROR(cufftPlan1d(&plan , nt2::numel(in) , CUFFT_R2C , 1));

      CUDA_ERROR(cufftExecR2C( plan
                  ,  in.data()
                  , (cufftDoubleComplex *) out.data()
                  , CUFFT_FORWARD
                  ));

      CUDA_ERROR(cufftDestroy(p));
    }
  };

  BOOST_DISPATCH_IMPLEMENT  ( fft_, nt2::tag::cuda_<site>
                            , (A0)(S0)(A1)(S1)(site)
                            , ((container_< nt2::tag::table_, complex_<double_<A0>>, S0 >))
                              ((container_< nt2::tag::table_, complex_<double_<A1>>, S1 >))
                            )
  {
    typedef void result_type;
    typedef typename A0::value_type s_type;

    BOOST_FORCEINLINE result_type operator()(A0 const& in, A1& out) const
    {
      out.resize(nt2::extent(in));

      cufftHandle plan;

      CUDA_ERROR(cufftPlan1d(&plan , nt2::numel(in) , CUFFT_Z2Z , 1));

      CUDA_ERROR(cufftExecZ2Z( plan
                  , (cufftDoubleComplex *) in.data()
                  , (cufftDoubleComplex *) out.data()
                  , CUFFT_FORWARD
                  ));

      CUDA_ERROR(cufftDestroy(p));
    }
  };

} }

#endif
#endif
