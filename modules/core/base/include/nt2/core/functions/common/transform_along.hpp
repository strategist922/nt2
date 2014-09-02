//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_EXPR_TRANSFORM_ALONG_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_EXPR_TRANSFORM_ALONG_HPP_INCLUDED

#include <nt2/core/functions/transform_along.hpp>
#include <nt2/core/container/dsl/forward.hpp>
#include <nt2/signal/details/conv1d.hpp>
#include <nt2/include/functions/numel.hpp>
#include <nt2/include/functions/run.hpp>

namespace nt2 { namespace ext
{
  //============================================================================
  // Global version
  //============================================================================
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::transform_along_, tag::cpu_
                            , (Out)(In)(K)(Rng)
                            , ((ast_<Out, nt2::container::domain>))
                              ((ast_<In, nt2::container::domain>))
                              (unspecified_<K>)
                              (unspecified_<Rng>)
                            )
  {
    typedef void result_type;

    result_type operator()( Out& out
                          , In& in, K const& kernel
                          , Rng const& r
                          ) const
    {
      int m  = numel(in);
      int n  = numel(kernel.extent());
      int n1 = n-1;
      int k=0, ok = r.second;

      typedef typename Out::value_type  out_t;

      // Prologue : Slide filter in
      int p = n1-ok-1;
      for(;k<=p;k++,ok++)
      {
        nt2::run(out,k,details::conv1D<out_t>(0,ok+1,ok+1,in,kernel));
      }

      // Center : Use all n stuff all the time up to the epilogue
      int e = std::min(m-1-ok,int(r.first-1));
      for(;k<=e;k++,ok++)
      {
        nt2::run(out,k,details::conv1D<out_t>(ok-n1,n,numel(kernel.extent()),in,kernel));
      }

      // Epilogue, slide down the end
      for(;k!=r.first;k++,ok++)
      {
        nt2::run(out,k,details::conv1D<out_t>(ok-n1,n,std::min(n,m+n1-ok),in,kernel));
      }
    }
  };
} }

#endif
