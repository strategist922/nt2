//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SIMD_TRANSFORM_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SIMD_TRANSFORM_HPP_INCLUDED
#ifndef BOOST_SIMD_NO_SIMD

#include <nt2/core/functions/transform.hpp>
#include <nt2/include/functions/run.hpp>
#include <nt2/include/functions/splat.hpp>
#include <nt2/include/functions/scalar/numel.hpp>
#include <boost/simd/sdk/simd/native.hpp>
#include <boost/simd/sdk/meta/cardinal_of.hpp>
#include <boost/simd/sdk/simd/meta/is_vectorizable.hpp>
#include <boost/fusion/include/pop_front.hpp>

#include <nt2/include/constants/valmax.hpp>

namespace nt2 { namespace ext
{
  //============================================================================
  // Global nD element-wise transform
  //============================================================================
  NT2_FUNCTOR_IMPLEMENTATION( nt2::tag::transform_, boost::simd::tag::simd_
                            , (A0)(A1)
                            , ((ast_<A0, nt2::container::domain>))
                              ((ast_<A1, nt2::container::domain>))
                            )
  {
    typedef void result_type;

    BOOST_FORCEINLINE result_type operator()(A0& a0, A1& a1) const
    {
      nt2::transform(a0,a1,0,nt2::numel(a0));
    }
  };

  template<class Tag, std::size_t N, class Expr>
  struct max_vect_size_impl;

  template<class Expr>
  BOOST_FORCEINLINE std::size_t max_vect_size(Expr const& expr)
  {
    return max_vect_size_impl<typename Expr::proto_tag, Expr::proto_arity_c, Expr const>()(expr);
  }

  template<class Tag, std::size_t N, class Expr>
  struct max_vect_size_impl2;

  template<class Tag, std::size_t N, class Expr>
  struct max_vect_size_impl : max_vect_size_impl2<Tag, N, Expr>
  {
  };

  template<class Tag, class Expr>
  struct max_vect_size_impl2<Tag, 0, Expr>
  {
    typedef std::size_t result_type;
    BOOST_FORCEINLINE result_type operator()(Expr& expr) const
    {
      return Valmax<result_type>();
    }
  };

  template<class Tag, class Expr>
  struct max_vect_size_impl2<Tag, 1, Expr>
  {
    typedef std::size_t result_type;
    BOOST_FORCEINLINE result_type operator()(Expr& expr) const
    {
      return max_vect_size(boost::proto::child_c<0>(expr));
    }
  };

  template<class Tag, class Expr>
  struct max_vect_size_impl2<Tag, 2, Expr>
  {
    typedef std::size_t result_type;
    BOOST_FORCEINLINE result_type operator()(Expr& expr) const
    {
      return std::min( max_vect_size(boost::proto::child_c<0>(expr))
                     , max_vect_size(boost::proto::child_c<1>(expr))
                     );
    }
  };

  template<class Tag, class Expr>
  struct max_vect_size_impl2<Tag, 3, Expr>
  {
    typedef std::size_t result_type;
    BOOST_FORCEINLINE result_type operator()(Expr& expr) const
    {
      return std::min( max_vect_size(boost::proto::child_c<0>(expr))
                     , std::min( max_vect_size(boost::proto::child_c<1>(expr))
                               , max_vect_size(boost::proto::child_c<2>(expr))
                               )
                     );
    }
  };

  template<class Tag, class Expr>
  struct max_vect_size_impl2<Tag, 4, Expr>
  {
    typedef std::size_t result_type;
    BOOST_FORCEINLINE result_type operator()(Expr& expr) const
    {
      return std::min( max_vect_size(boost::proto::child_c<0>(expr))
                     , std::min( max_vect_size(boost::proto::child_c<1>(expr))
                               , std::min( max_vect_size(boost::proto::child_c<2>(expr))
                                         , max_vect_size(boost::proto::child_c<3>(expr))
                                         )
                               )
                     );
    }
  };

  //============================================================================
  // Partial nD element-wise transform with offset/size
  //============================================================================
  NT2_FUNCTOR_IMPLEMENTATION_IF( nt2::tag::transform_, boost::simd::tag::simd_
                               , (A0)(A1)(A2)(A3)
                               , (boost::simd::meta::is_vectorizable<typename A0::value_type, BOOST_SIMD_DEFAULT_EXTENSION>)
                               , ((ast_<A0, nt2::container::domain>))
                                 ((ast_<A1, nt2::container::domain>))
                                 (scalar_< integer_<A2> >)
                                 (scalar_< integer_<A3> >)
                               )
  {
    typedef void result_type;

    typedef typename A0::value_type stype;
    typedef boost::simd::native<stype, BOOST_SIMD_DEFAULT_EXTENSION> target_type;

    BOOST_FORCEINLINE result_type
    operator()(A0& a0, A1& a1, A2 p, A3 sz) const
    {
      std::size_t inner_sz = std::min(max_vect_size(a0), max_vect_size(a1));
      inner_sz = std::min(inner_sz, sz);

      static const std::size_t N = boost::simd::meta
                                        ::cardinal_of<target_type>::value;

      std::size_t aligned_sz  = inner_sz & ~(N-1);
      std::size_t it          = p;

      for(std::size_t p2 = it; it != p+sz; p2 = it)
      {
        for(std::size_t m=p2+aligned_sz; it != m; it+=N)
          nt2::run( a0, it, nt2::run(a1, it, meta::as_<target_type>()) );

        for(std::size_t m=p2+inner_sz; it != m; ++it)
          nt2::run( a0, it, nt2::run(a1, it, meta::as_<stype>()) );
      }
    }
  };
} }

#endif
#endif
