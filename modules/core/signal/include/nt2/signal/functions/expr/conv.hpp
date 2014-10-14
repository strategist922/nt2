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
#include <nt2/signal/details/conv_size.hpp>
#include <nt2/signal/details/conv_offset.hpp>
#include <nt2/include/functions/transform_along.hpp>
#include <nt2/include/functions/isvector.hpp>
#include <nt2/include/functions/run.hpp>
#include <nt2/signal/details/as_stencil.hpp>
#include <boost/assert.hpp>
#include <nt2/core/container/dsl/as_terminal.hpp>

namespace nt2 { namespace ext
{
  // Building conv node from AST correlator
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::conv_, tag::cpu_
                            , (A0)(A1)(Shp)
                            , ((ast_<A0, nt2::container::domain>))
                              ((ast_<A1, nt2::container::domain>))
                              (unspecified_<Shp>)
                            )
  {
    typedef typename details::conv_size<Shp,A0,A1>::result_type       sh_t;

    typedef typename  boost::proto::
                      result_of::make_expr< nt2::tag::conv_
                                          , container::domain
                                          , A0 const&
                                          , A1 const&
                                          , Shp const&
                                          , sh_t
                                          , boost::mpl::true_
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
                                      , details::conv_size<Shp,A0,A1>::call(a0,a1)
                                      , boost::mpl::true_()
                                      );
    }
  };

  // Building conv node from ConvolutionOperator
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::conv_, tag::cpu_
                            , (A0)(A1)(Shp)
                            , ((ast_<A0, nt2::container::domain>))
                              (unspecified_<A1>)
                              (unspecified_<Shp>)
                            )
  {
    typedef typename details::conv_size<Shp,A0,A1>::result_type       sh_t;

    typedef typename  boost::proto::
                      result_of::make_expr< nt2::tag::conv_
                                          , container::domain
                                          , A0 const&
                                          , A1 const&
                                          , Shp const&
                                          , sh_t
                                          , boost::mpl::false_
                                          >::type             result_type;

    BOOST_FORCEINLINE
    result_type operator()(A0 const& a0, A1 const& a1, Shp const& shp) const
    {
      BOOST_ASSERT_MSG( isvector(a0) && a0.size()
                      , "Error in conv - First parameter must be a vector"
                      );

      return boost::proto::make_expr< nt2::tag::conv_
                                    , container::domain
                                    > ( boost::cref(a0)
                                      , boost::cref(a1)
                                      , boost::cref(shp)
                                      , details::conv_size<Shp,A0,A1>::call(a0,a1)
                                      , boost::mpl::false_()
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
    typedef A0& result_type;

    typedef typename boost::proto::result_of::child_c<A1&,2>::value_type  c2_t;
    typedef typename boost::proto::result_of::value<c2_t>::value_type     mode_t;

    BOOST_FORCEINLINE result_type operator()(A0& out, const A1& in) const
    {

      out.resize(in.extent());
      std::size_t co = conv_offset(boost::proto::child_c<1>(in), mode_t());

      // TODO: Find better discrimination on the status of a1

      return eval(out,in,co, boost::proto::value(boost::proto::child_c<4>(in)));

    }

    // Require as_filter adaptation
    BOOST_FORCEINLINE result_type
    eval(A0& out, const A1& in, std::size_t co, boost::mpl::true_ const&) const
    {
      if(boost::proto::child_c<0>(in).size() >= boost::proto::child_c<1>(in).size())
      {
        transform_along ( out
                        , boost::proto::child_c<0>(in)
                        , details::as_stencil( boost::proto::child_c<1>(in) )
                        , boost::proto::value(boost::proto::child_c<2>(in))
                        );
      }
      else
      {
        transform_along ( out
                        , boost::proto::child_c<1>(in)
                        , details::as_stencil( boost::proto::child_c<0>(in) )
                        , boost::proto::value(boost::proto::child_c<2>(in))
                        );
      }

          return out;
    }

    // a1 is already a ConvolutionOperator
    BOOST_FORCEINLINE result_type
    eval(A0& out, const A1& in, std::size_t co, boost::mpl::false_ const&) const
    {
      transform_along ( out
                      , boost::proto::child_c<0>(in)
                      , boost::proto::value(boost::proto::child_c<1>(in))
                      , boost::proto::value(boost::proto::child_c<2>(in))
                      );

      return out;
    }
  };
} }

#endif
