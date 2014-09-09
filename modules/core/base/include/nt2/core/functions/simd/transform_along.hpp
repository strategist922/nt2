//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SIMD_TRANSFORM_ALONG_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SIMD_TRANSFORM_ALONG_HPP_INCLUDED

#include <nt2/core/functions/transform_along.hpp>
#include <nt2/core/container/dsl/forward.hpp>
#include <nt2/signal/details/conv1d.hpp>
#include <nt2/include/functions/run.hpp>
#include <nt2/sdk/meta/type_id.hpp>
#include <boost/simd/sdk/meta/cardinal_of.hpp>

namespace nt2 { namespace ext
{
  //============================================================================
  // Global version
  //============================================================================
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::transform_along_, boost::simd::tag::simd_
                            , (Out)(In)(K)(O)
                            , ((ast_<Out, nt2::container::domain>))
                              ((ast_<In, nt2::container::domain>))
                              (unspecified_<K>)
                              (scalar_< integer_<O> >)
                            )
  {
    typedef void result_type;

    result_type operator()( Out& out, In& in, K const& kernel, O offset ) const
    {
      transform_along ( out, in, kernel, offset
                      , std::make_pair( 0, out.size() )
                      );
    }
  };

  //============================================================================
  // Ranged version
  //============================================================================
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::transform_along_, boost::simd::tag::simd_
                            , (Out)(In)(K)(Rng)(O)
                            , ((ast_<Out, nt2::container::domain>))
                              ((ast_<In, nt2::container::domain>))
                              (unspecified_<K>)
                              (scalar_< integer_<O> >)
                              (unspecified_<Rng>)
                            )
  {
    typedef void result_type;

    result_type operator()( Out& out
                          , In& in, K const& kernel
                          , O offset
                          , Rng const& r
                          ) const
    {
      int m  = in.size();
      int n  = kernel.size();
      int n1 = n-1;
      int k=r.first, ok = k+offset;
      int end = r.first+r.second;

      typedef typename Out::value_type  out_t;
      typedef typename In::value_type in_t;
      typedef boost::simd::native<out_t, BOOST_SIMD_DEFAULT_EXTENSION> target_type;

      std::size_t card = boost::simd::meta::cardinal_of<target_type>::value;

      //Prologue : Slide filter in
      int p = n1-ok-1;

      for(;k<=p;k++,ok++)
      {
        //TODO: Find a formula keeping static numel in
        nt2::run(out,k,details::conv1D<out_t>(0,ok+1,ok+1,in,kernel));
      }

      //Center : Use all n stuff all the time up to the epilogue
      int e = std::min(m-1-ok,int(end-1));
      std::size_t aligned_sz  = std::max((e-k),0) & ~(card-1);
      std::size_t mm = k + aligned_sz;

      for(; k + card < mm; k += card)
        {
          //entry_type in_data = boost::simd::load< entry_type >(&in.raw()[k]);
          nt2::run(out,k,details::conv1D<target_type >(ok-n1,n,kernel.size(),in,kernel));
        }

      for(;k<=e;k++,ok++)
        {
         // we pass numel instead of n to keep the static informations for unrolling
          nt2::run(out,k,details::conv1D<out_t>(ok-n1,n,kernel.size(),in,kernel));
        }

      // Epilogue, slide down the end
      for(;k!=end;k++,ok++)
        {
          //TODO: Find a formula keeping static numel in
          nt2::run(out,k,details::conv1D<out_t>(ok-n1,n,std::min(n,m+n1-ok),in,kernel));
        }
    }
  };
} }

#endif
