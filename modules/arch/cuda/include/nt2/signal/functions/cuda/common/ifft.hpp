//==============================================================================
//         Copyright 2015 NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SIGNAL_FUNCTIONS_CUDA_COMMON_FFT_HPP_INCLUDED
#define NT2_SIGNAL_FUNCTIONS_CUDA_COMMON_FFT_HPP_INCLUDED

#if defined(NT2_HAS_CUDA) && defined(NT2_USE_CUFFTW)

#include <nt2/sdk/cuda/cuda.hpp>
#include <nt2/signal/functions/ifft.hpp>
#include <nt2/include/functions/numel.hpp>
#include <nt2/core/container/dsl/as_terminal.hpp>
#include <nt2/core/utility/assign_swap.hpp>
#include <nt2/sdk/meta/device.hpp>
#include <cuda_runtime.h>
#include <cufft.h>

namespace nt2 { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT  ( ifft_, nt2::tag::cuda_<site>
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

      auto device_in = to_device(in);
      auto device_out = to_device(out);

      cufftHandle plan;

      cufftPlan1d(&plan , nt2::numel(in) , CUFFT_R2C , 1);

      cufftExecR2C( plan
                  ,  const_cast<cufftReal*> (device_in.data())
                  , (cufftComplex *) device_out.data()
                  );

      cufftDestroy(plan);

      device_swap(device_out,out);
    }
  };

  BOOST_DISPATCH_IMPLEMENT  ( ifft_, nt2::tag::cuda_<site>
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

      auto device_in = to_device(in);
      auto device_out = to_device(out);

      cufftHandle plan;

      cufftPlan1d(&plan , nt2::numel(in) , CUFFT_C2C , 1);

      cufftExecC2C( plan
                  ,  const_cast<cufftComplex *>( device_in.data() )
                  , (cufftComplex *) device_out.data()
                  , CUFFT_INVERSE
                  );

      cufftDestroy(plan);

      device_swap(device_out,out);
    }
  };

  BOOST_DISPATCH_IMPLEMENT  ( ifft_, nt2::tag::cuda_<site>
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

      auto device_in = to_device(in);
      auto device_out = to_device(out);

      cufftHandle plan;

      cufftPlan1d(&plan , nt2::numel(in) , CUFFT_D2Z , 1);

      cufftExecD2Z( plan
                  , const_cast<cufftDoubleReal *> (device_in.data())
                  , (cufftDoubleComplex *) device_out.data()
                  );

      cufftDestroy(plan);

      device_swap(device_out,out);
    }
  };

  BOOST_DISPATCH_IMPLEMENT  ( ifft_, nt2::tag::cuda_<site>
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

      auto device_in = to_device(in);
      auto device_out = to_device(out);

      cufftHandle plan;

      cufftPlan1d(&plan , nt2::numel(in) , CUFFT_Z2Z , 1);

      cufftExecZ2Z( plan
                  ,  const_cast<cufftDoubleComplex *>( device_in.data() )
                  , (cufftDoubleComplex *) device_out.data()
                  , CUFFT_INVERSE
                  );

      cufftDestroy(plan);

      device_swap(device_out,out);
    }
  };

} }

#endif
#endif
