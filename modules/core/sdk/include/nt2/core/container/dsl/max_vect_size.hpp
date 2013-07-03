//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_CONTAINER_DSL_MAX_VECT_SIZE_HPP_INCLUDED
#define NT2_CORE_CONTAINER_DSL_MAX_VECT_SIZE_HPP_INCLUDED

#include <nt2/include/constants/valmax.hpp>
#include <boost/proto/traits.hpp>
#include <algorithm>

namespace nt2 { namespace ext
{
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
} }

#endif
