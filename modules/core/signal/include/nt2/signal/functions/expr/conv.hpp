//==============================================================================
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SIGNAL_FUNCTIONS_EXPR_CONV_HPP_INCLUDED
#define NT2_SIGNAL_FUNCTIONS_EXPR_CONV_HPP_INCLUDED

#include <nt2/signal/options.hpp>
#include <nt2/signal/functions/conv.hpp>
#include <nt2/signal/details/as_filter.hpp>
#include <nt2/signal/details/conv_offset.hpp>
#include <nt2/include/functions/transform_along.hpp>
#include <nt2/include/functions/isvector.hpp>
#include <nt2/include/functions/run.hpp>
#include <boost/assert.hpp>

namespace nt2 { namespace ext
{
  // Building conv node
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::conv_, tag::cpu_
                            , (A0)(A1)(Shp)
                            , ((ast_<A0, nt2::container::domain>))
                              ((ast_<A1, nt2::container::domain>))
                              (unspecified_<Shp>)
                            )
  {
    template<typename Shape, typename Dummy = void> struct make_size
    {
      typedef _2D result_type;

      static BOOST_FORCEINLINE result_type call(A0 const& a0, A1 const& a1)
      {
        // We use numel(a1.extent()) as extent() is define din both
        // Expression and ConvolutionOperator concept
        std::ptrdiff_t n = a0.size()+a1.size()-1;
        return a0.extent()[0] == 1 ? result_type(1,n) : result_type(n);
      }
    };

    template<typename Dummy> struct make_size< policy<ext::same_>, Dummy >
    {
      typedef typename A0::extent_type                      result_type;

      static BOOST_FORCEINLINE result_type call(A0 const& a0, A1 const& a1)
      {
        return a0.extent();
      }
    };

    template<typename Dummy> struct make_size< policy<ext::valid_>, Dummy >
    {
      typedef _2D result_type;

      static BOOST_FORCEINLINE result_type call(A0 const& a0, A1 const& a1)
      {
        std::ptrdiff_t l0 = a0.size();
        std::ptrdiff_t l1 = a1.size();

        std::ptrdiff_t n  = std::max( l0 - std::max(std::ptrdiff_t(0), l1-1)
                                    , std::ptrdiff_t(0)
                                    );

        return a0.extent()[0] == 1 ? result_type(1,n) : result_type(n);
      }
    };

    typedef typename make_size<Shp>::result_type       sh_t;

    typedef typename  boost::proto::
                      result_of::make_expr< nt2::tag::conv_
                                          , container::domain
                                          , A0 const&
                                          , A1 const&
                                          , Shp const&
                                          , sh_t
                                          >::type             result_type;

    BOOST_FORCEINLINE
    result_type operator()(A0 const& a0, A1 const& a1, Shp const& shp) const
    {
      BOOST_ASSERT_MSG( isvector(a0) && a0.size()
                      , "Error in conv - First parameter must be a vector"
                      );

      BOOST_ASSERT_MSG( isvector(a1) && a1.size()
                      , "Error in conv - Second parameter must be a vector"
                      );

      return boost::proto::make_expr< nt2::tag::conv_
                                    , container::domain
                                    > ( boost::cref(a0)
                                      , boost::cref(a1)
                                      , boost::cref(shp)
                                      , make_size<Shp>::call(a0,a1)
                                      );
    }
  };

  // Evaluation of x = conv(u,v)
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::run_assign_, tag::cpu_
                            , (A0)(A1)(N)
                            , ((ast_<A0, nt2::container::domain>))
                              ((node_<A1,nt2::tag::conv_,N,nt2::container::domain>))
                            )
  {
    typedef A0&             result_type;

    typedef typename boost::proto::result_of::child_c<A1&,2>::value_type  c2_t;
    typedef typename boost::proto::result_of::value<c2_t>::value_type     mode_t;

    BOOST_FORCEINLINE result_type operator()(A0& out, const A1& in) const
    {
      out.resize(in.extent());
      std::size_t co = conv_offset(boost::proto::child_c<1>(in), mode_t());

      if(boost::proto::child_c<0>(in).size() >= boost::proto::child_c<1>(in).size())
      {
        transform_along( out
                        , boost::proto::child_c<0>(in)
                        , as_filter( boost::proto::child_c<1>(in) )
                        , std::make_pair( out.size(), co )
                        );
      }
      else
      {
        transform_along( out
                        , boost::proto::child_c<1>(in)
                        , as_filter( boost::proto::child_c<0>(in) )
                        , std::make_pair( out.size(), co )
                        );
      }

      return out;
    }
  };
} }

#endif
