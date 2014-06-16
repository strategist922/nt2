//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_CONTAINER_VIEW_SHARED_VIEW_HPP_INCLUDED
#define NT2_CORE_CONTAINER_VIEW_SHARED_VIEW_HPP_INCLUDED

#include <nt2/sdk/memory/container_ref.hpp>
#include <nt2/sdk/memory/container_shared_ref.hpp>
#include <nt2/core/container/view/adapted/shared_view.hpp>
#include <nt2/core/container/view/adapted/shared_view_type.hpp>
#include <boost/dispatch/dsl/semantic_of.hpp>

namespace nt2 { namespace container
{
  /* shared_view; an expression of a container_shared_ref terminal.
   * allows construction from an expression of a container_shared_ref terminal */
  template<typename Container>
  struct  shared_view
        : details::shared_view_type<Container>::nt2_expression
  {
    typedef details::shared_view_type<Container>    sv_t;
    typedef typename sv_t::basic_expr               basic_expr;
    typedef typename sv_t::container_ref            container_ref;
    typedef typename sv_t::container_type           container_type;
    typedef typename sv_t::nt2_expression           nt2_expression;
    typedef typename sv_t::value_type               value_type;

    typedef typename container_ref::iterator        iterator;
    typedef typename container_ref::const_iterator  const_iterator;

    iterator begin()  const { return boost::proto::value(*this).begin(); }
    iterator end()    const { return boost::proto::value(*this).end();   }

    BOOST_FORCEINLINE
    shared_view()
    {
    }

    BOOST_FORCEINLINE
    shared_view( nt2_expression const& expr )
                     : nt2_expression(expr)
    {
    }

    template<class Xpr>
    BOOST_FORCEINLINE
    shared_view( Xpr const& expr )
                     : nt2_expression(basic_expr::make(boost::proto::value(expr)))
    {
    }

    template<class Xpr>
    void reset(Xpr const& other)
    {
      shared_view tmp(other);
      boost::proto::value(*this) = boost::proto::value(tmp);
      this->size_ = tmp.size_;
    }


    //==========================================================================
    // Enable base expression handling of assignment
    //==========================================================================
    template<class Xpr> BOOST_FORCEINLINE
    typename boost::disable_if< boost::is_base_of<nt2_expression, Xpr>
                              , shared_view&
                              >::type
    operator=(Xpr const& xpr)
    {
      nt2_expression::operator=(xpr);
      return *this;
    }

    BOOST_FORCEINLINE
    shared_view& operator=(shared_view const& xpr)
    {
      nt2_expression::operator=(xpr);
      return *this;
    }

    template<class Xpr> BOOST_FORCEINLINE
    typename boost::disable_if< boost::is_base_of<nt2_expression, Xpr>
                              , shared_view const&
                              >::type
    operator=(Xpr const& xpr) const
    {
      nt2_expression::operator=(xpr);
      return *this;
    }

    BOOST_FORCEINLINE
    shared_view const& operator=(shared_view const& xpr) const
    {
      nt2_expression::operator=(xpr);
      return *this;
    }
  };

  template<typename Container>
  struct  shared_view<Container const>
        : details::shared_view_type<Container const>::nt2_expression
  {
    typedef details::shared_view_type<Container const>  sv_t;
    typedef typename sv_t::basic_expr                   basic_expr;
    typedef typename sv_t::container_ref                container_ref;
    typedef typename sv_t::container_type               container_type;
    typedef typename sv_t::nt2_expression               nt2_expression;
    typedef typename sv_t::value_type                   value_type;

    typedef typename container_ref::iterator            iterator;
    typedef typename container_ref::const_iterator      const_iterator;

    iterator begin()  const { return boost::proto::value(*this).begin(); }
    iterator end()    const { return boost::proto::value(*this).end();   }

    BOOST_FORCEINLINE
    shared_view()
    {
    }

    BOOST_FORCEINLINE
    shared_view( nt2_expression const& expr )
                     : nt2_expression(expr)
    {
    }

    template<class Xpr>
    shared_view( Xpr const& expr )
                     : nt2_expression(basic_expr::make(boost::proto::value(expr)))
    {
    }

    template<class Xpr>
    BOOST_FORCEINLINE
    void reset(Xpr const& other)
    {
      shared_view tmp(other);
      boost::proto::value(*this) = boost::proto::value(tmp);
      this->size_ = tmp.size_;
    }

    //==========================================================================
    // Enable base expression handling of assignment
    //==========================================================================
    template<class Xpr> BOOST_FORCEINLINE
    typename boost::disable_if< boost::is_base_of<nt2_expression, Xpr>
                              , shared_view&
                              >::type
    operator=(Xpr const& xpr)
    {
      nt2_expression::operator=(xpr);
      return *this;
    }

    BOOST_FORCEINLINE
    shared_view& operator=(shared_view const& xpr)
    {
      nt2_expression::operator=(xpr);
      return *this;
    }

    template<class Xpr> BOOST_FORCEINLINE
    typename boost::disable_if< boost::is_base_of<nt2_expression, Xpr>
                              , shared_view const&
                              >::type
    operator=(Xpr const& xpr) const
    {
      nt2_expression::operator=(xpr);
      return *this;
    }

    BOOST_FORCEINLINE
    shared_view const& operator=(shared_view const& xpr) const
    {
      nt2_expression::operator=(xpr);
      return *this;
    }
  };
} }

namespace nt2
{
  using nt2::container::shared_view;
}

#endif
