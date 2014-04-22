//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2014   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef FILTERS_APPLY_STENCIL_HPP_INCLUDED
#define FILTERS_APPLY_STENCIL_HPP_INCLUDED

#include <cstddef>
#include <vector>
#include "window.hpp"
#include "reduction.hpp"
#include <boost/dispatch/meta/strip.hpp>
#include <boost/simd/sdk/meta/as_pack.hpp>
#include <boost/simd/include/functions/simd/aligned_store.hpp>

namespace nt2
{
  namespace details
  {
    // Regular stencil : don't have any optimization opportunities
    template< std::size_t Card
            , typename Operation
            , typename Input, typename Output
            >
    void apply_stencil( Operation o
                      , Input const& in, Output const& out
                      , std::size_t h, std::size_t w
                      , nt2::tag::regular_stencil_ const&
                      )
    {
      using boost::simd::aligned_store;

      typedef typename Input::value_type                                value_t;
      typedef typename boost::remove_const< typename boost::remove_pointer<value_t>::type >::type base_type;
      typedef typename boost::simd::meta::as_pack<base_type,Card>::type    type;

      static const std::size_t hh = Operation::height/2;
      static const std::size_t hw = (Operation::width/2)*Card;

      for(std::size_t j = hh; j<h-hh; ++j)
      {
        for(std::size_t i = hw; i<w-hw; i+=Card)
        {
          nt2::window<Operation,type> pxs(j,i,in);
          aligned_store( pxs.fold(), &out[j][i] );
        }
      }
    }

    // Regular stencil : don't have any optimization opportunities
    template< std::size_t Card
            , typename Operation
            , typename Input, typename Output
            >
    void apply_stencil( Operation o
                      , Input const& in, Output const& out
                      , std::size_t h, std::size_t w
                      , nt2::tag::reductible_stencil_ const&
                      )
    {
      using boost::simd::aligned_store;

      typedef typename Input::value_type                                value_t;
      typedef typename boost::remove_const< typename boost::remove_pointer<value_t>::type >::type base_type;
      typedef typename boost::simd::meta::as_pack<base_type,Card>::type    type;

      typedef typename
              Operation::template rebind<Operation::height,1>::other h_op;

      static const std::size_t hh = Operation::height/2;
      static const std::size_t hw = (Operation::width/2)*Card;

      nt2::window<h_op,type> rs;

      for(std::size_t j = hh; j<h-hh; ++j)
      {
        details::reducer_<Operation::width-1,type>::call(j,rs,in);

        for(std::size_t i = hw; i<w-hw; i+=Card)
        {
          details::reduce_column<Operation,type>(j, i+1, rs[Operation::width-1], in);

          rs.slide();
          aligned_store( rs.fold(), &out[j][i] );

          rs.shift();
        }
      }
    }
  }

  // Apply a stencil operation on a bunch of data
  template< std::size_t Card  // DEMO PURPOSE ONLY
          , typename Operation
          , typename Input, typename Output
          >
  void apply_stencil( Operation o
                    , Input const& din, Output& dout
                    , std::size_t h, std::size_t w
                    )
  {
    typedef typename Input::value_type itype;
    typedef typename Output::value_type otype;

    // Turn into local NRC-access data
    std::vector<itype const*> in;
    std::vector<otype*> out;

    in.resize(h);
    out.resize(h);

    in[0]  = &din[0];
    out[0] = &dout[0];

    for(std::size_t i = 1; i<h; ++i)
    {
      in[i]  = in [i-1] + w;
      out[i] = out[i-1] + w;
    }

    details::apply_stencil<Card>(o,in,out,h,w,typename Operation::filter_tag());
  }
}

#endif
