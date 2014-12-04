//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SHARED_MEMORY_TRANSFORM_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SHARED_MEMORY_TRANSFORM_HPP_INCLUDED

#include <nt2/core/functions/transform.hpp>
#include <nt2/sdk/shared_memory.hpp>
#include <nt2/sdk/shared_memory/worker/transform.hpp>
#include <nt2/sdk/shared_memory/spawner.hpp>
#include <nt2/sdk/config/cache.hpp>

#include <cstdio>
#include <utility>

namespace nt2 { namespace ext
{
  //============================================================================
  // Partial Shared Memory elementwise operation
  // Generates a SPMD loop nest and forward to internal site for evaluation
  // using the partial transform syntax.
  //============================================================================
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::transform_, (nt2::tag::shared_memory_<BackEnd,Site>)
                            , (Out)(In)(BackEnd)(Site)(Range)
                            , ((ast_<Out, nt2::container::domain>))
                              ((ast_<In, nt2::container::domain>))
                              (unspecified_<Range>)
                            )
  {

    typedef void result_type;
    typedef typename boost::remove_reference<In>::type::extent_type extent_type;

    BOOST_FORCEINLINE result_type operator()(Out& out, In& in, Range range) const
    {
      // std::size_t cache  = 1024*config::top_cache_size(1)/sizeof(typename Out::value_type);
      // std::size_t grain  = 800*800;
       std::size_t grain  = 3000*3000;

       std::size_t begin = range.first;
       std::size_t size = range.second;

       if(!grain) grain = 1u;

       nt2::worker<tag::transform_,BackEnd,Site,Out,In> w(out,in);

       nt2::spawner<tag::transform_,tag::asynchronous_<BackEnd> > s;
       s(w,begin,size,std::make_pair(grain,w.obound_));

       // nt2::spawner<tag::transform_,BackEnd > s;
       // s(w,begin,size,grain);
    }
  };

} }
#endif
