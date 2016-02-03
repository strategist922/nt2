//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_EXPR_EXTENT_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_EXPR_EXTENT_HPP_INCLUDED

#include <nt2/core/functions/extent.hpp>
#include <nt2/core/utility/of_size.hpp>
#include <nt2/core/container/dsl/forward.hpp>
#include <boost/fusion/include/transform.hpp>
#include <boost/fusion/include/fold.hpp>
#include <boost/proto/fusion.hpp>
#include <boost/mpl/long.hpp>

namespace nt2 { namespace ext
{
  BOOST_DISPATCH_IMPLEMENT  ( extent_, tag::cpu_
                            , (A)(T)
                            , ((node_ < A, elementwise_<T>
                                      , boost::mpl::long_<0>
                                      , nt2::container::domain
                                      >
                              ))
                            )
  {
    BOOST_DISPATCH_RETURNS(1, (A const& a0),nt2::extent(boost::proto::value(a0)));
  };

  BOOST_DISPATCH_IMPLEMENT( extent_, tag::cpu_
                          , (A)(T)
                          , ((node_ < A
                                    , elementwise_<T>
                                    , boost::mpl::long_<1>
                                    , nt2::container::domain
                                    >
                            ))
                          )
  {
    BOOST_DISPATCH_RETURNS(1, (A const& a0), nt2::extent(boost::proto::child_c<0>(a0)));
  };

  BOOST_DISPATCH_IMPLEMENT( extent_, tag::cpu_
                          , (A)(T)(N)
                          , ((node_ < A
                                    , elementwise_<T>
                                    , N
                                    , nt2::container::domain
                                    >
                            ))
                          )
  {
    // TODO: [C++14] Remove this and use polymorphic lambda
    struct get_extent
    {
      template<typename X>
      BOOST_FORCEINLINE auto operator()(X const& x) const -> decltype(nt2::extent(x))
      {
        return nt2::extent(x);
      }
    };

    // TODO: [C++14] Remove this and use polymorphic lambda
    struct size_fold
    {
      template<typename A1>
      BOOST_FORCEINLINE A1 const& operator()(_0D const&, A1 const& a1) const
      {
        return a1;
      }

      template<typename A0>
      BOOST_FORCEINLINE A0 const& operator()(A0 const& a0, _0D const&) const
      {
        return a0;
      }

      BOOST_FORCEINLINE _0D operator()(_0D const&, _0D const&) const
      {
        return {};
      }

      template<typename A0, typename A1> static BOOST_FORCEINLINE
      A0 const& selection(A0 const& a0, A1 const&, boost::mpl::true_ const&)
      {
        return a0;
      }

      template<typename A0, typename A1> static BOOST_FORCEINLINE
      A1 const& selection(A0 const&, A1 const& a1, boost::mpl::false_ const&)
      {
        return a1;
      }

      template<typename A0, typename A1> BOOST_FORCEINLINE
      auto operator()(A0 const& a0, A1 const& a1) const
        -> decltype(selection(a0,a1,boost::mpl::bool_<A0::static_status>()))
      {
        BOOST_ASSERT_MSG(a0 == a1, "Sizes are not compatible");
        return selection(a0,a1,boost::mpl::bool_<A0::static_status>());
      }
    };

   BOOST_DISPATCH_RETURNS ( 1, (A const& a0)
                          , boost::fusion
                                 ::fold ( boost::fusion::transform(a0,get_extent{})
                                        , get_extent{}(boost::proto::child_c<0>(a0))
                                        , size_fold{}
                                        )
                          );
  };
} }

#endif
