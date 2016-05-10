//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_CONTAINER_VIEW_VIEW_HPP_INCLUDED
#define NT2_CORE_CONTAINER_VIEW_VIEW_HPP_INCLUDED

#include <nt2/sdk/memory/container_ref.hpp>
#include <nt2/sdk/memory/container_shared_ref.hpp>
#include <nt2/core/container/view/adapted/view.hpp>
#include <nt2/core/container/view/adapted/view_type.hpp>
#include <boost/dispatch/dsl/semantic_of.hpp>
#include <boost/config.hpp>

#if defined(BOOST_MSVC)
#pragma warning( push )
#pragma warning( disable : 4522 ) // multiple assignment operators specified
#endif

namespace nt2 { namespace container
{
  /* view; an expression of a container_ref terminal.
   * allows construction from an expression of a container terminal */
  template<typename Container>
  struct view : details::view_type<Container>::nt2_expression
  {
    typedef details::view_type<Container>           v_t;
    typedef typename v_t::basic_expr                basic_expr;
    typedef typename v_t::container_ref             container_ref;
    typedef typename v_t::container_type            container_type;
    typedef typename v_t::nt2_expression            nt2_expression;
    typedef typename v_t::value_type                value_type;

    typedef typename container_ref::pointer         pointer;
    typedef typename container_ref::iterator        iterator;
    typedef typename container_ref::const_iterator  const_iterator;
    typedef typename container_ref::locality_t      locality_t;

    iterator begin()  const { return boost::proto::value(*this).begin(); }
    iterator end()    const { return boost::proto::value(*this).end(); }

    /// @brief Default constructor
    BOOST_FORCEINLINE
    view()
    {
    }

    /// @brief Constructor from existing expression
    BOOST_FORCEINLINE
    view( nt2_expression const& expr )
              : nt2_expression(expr)
    {
    }

    /// @brief Constructor from existing base_expr
    template<class Xpr>
    BOOST_FORCEINLINE
    view( Xpr& expr )
              : nt2_expression(basic_expr::make(boost::proto::value(expr)))
    {
    }

    /// @brief Constructor from existing constant base_expr
    template<class Xpr>
    BOOST_FORCEINLINE
    view( Xpr const& expr )
              : nt2_expression(basic_expr::make(boost::proto::value(expr)))
    {
    }

    /*!
      @brief View from pointer constructor

      Build a copy-less view from a pointer and a given size. Resulting view
      provide access to the underlying data block through the usual Container
      interface.
    **/
    template<typename Extent>
    BOOST_FORCEINLINE
    view(pointer p, Extent const& sz)
            : nt2_expression(basic_expr::make(container_ref(p, sz)))
    {
    }

    template<typename K, typename T, typename S>
    BOOST_FORCEINLINE
    view(memory::container<K, T, S>& c)
            : nt2_expression(basic_expr::make(container_ref(c)))
    {
    }

    template<typename K, typename T, typename S>
    BOOST_FORCEINLINE
    view(memory::container<K, T, S> const& c)
            : nt2_expression(basic_expr::make(container_ref(c)))
    {
    }

    template<class Xpr>
    BOOST_FORCEINLINE void reset(Xpr& other)
    {
      view tmp(other);
      boost::proto::value(*this) = boost::proto::value(tmp);
      this->size_ = tmp.size_;
    }

    template<typename Extent>
    BOOST_FORCEINLINE void reset(pointer p, Extent const& sz)
    {
      view tmp(p,sz);
      boost::proto::value(*this) = boost::proto::value(tmp);
      this->size_ = tmp.size_;
    }

    //==========================================================================
    // Enable base expression handling of assignment
    //==========================================================================
    template<class Xpr> BOOST_FORCEINLINE
    typename boost::disable_if< boost::is_base_of<nt2_expression, Xpr>
                              , view&
                              >::type
    operator=(Xpr const& xpr)
    {
      using check = boost::mpl::bool_< meta::is_device_assign<view,Xpr>::value
                                    && meta::is_container_and_terminal<Xpr>::value
                                     >;
      return eval(xpr,check{});
    }

    template<class Xpr> BOOST_FORCEINLINE
    view& eval(Xpr const& xpr , boost::mpl::true_ const&)
    {
        boost::proto::value(*this).assign(boost::proto::value(xpr));
        return *this;
    }

    template<class Xpr> BOOST_FORCEINLINE
    view& eval(Xpr const& xpr , boost::mpl::false_ const&)
    {
        nt2_expression::operator=(xpr);
        return *this;
    }

    template<class Xpr> BOOST_FORCEINLINE
    typename boost::disable_if< boost::is_base_of<nt2_expression, Xpr>
                              , view const&
                              >::type
    operator=(Xpr const& xpr) const
    {
      using check = boost::mpl::bool_< meta::is_device_assign<view,Xpr>::value
                                    && meta::is_container_and_terminal<Xpr>::value
                                     >;
      return eval(xpr,check{});
    }

    template<class Xpr> BOOST_FORCEINLINE
    view const& eval(Xpr const& xpr , boost::mpl::true_ const&) const
    {
        boost::proto::value(*this).assign(boost::proto::value(xpr));
        return *this;
    }

    template<class Xpr> BOOST_FORCEINLINE
    view const& eval(Xpr const& xpr , boost::mpl::false_ const&) const
    {
        nt2_expression::operator=(xpr);
        return *this;
    }


    BOOST_FORCEINLINE view const& operator=(view const& xpr) const
    {
      nt2_expression::operator=(xpr);
      return *this;
    }
  };

  template<typename Container>
  struct  view<Container const>
        : details::view_type<Container const>::nt2_expression
  {
    typedef details::view_type<Container const>     v_t;
    typedef typename v_t::basic_expr                basic_expr;
    typedef typename v_t::container_ref             container_ref;
    typedef typename v_t::container_type            container_type;
    typedef typename v_t::nt2_expression            nt2_expression;
    typedef typename v_t::value_type                value_type;

    typedef typename container_ref::iterator        iterator;
    typedef typename container_ref::const_iterator  const_iterator;

    iterator begin()  const { return boost::proto::value(*this).begin();  }
    iterator end()    const { return boost::proto::value(*this).end();    }

    BOOST_FORCEINLINE
    view()
    {
    }

    BOOST_FORCEINLINE
    view( nt2_expression const& expr )
              : nt2_expression(expr)
    {
    }

    template<class Xpr>
    BOOST_FORCEINLINE
    view( Xpr const& expr )
              : nt2_expression(basic_expr::make(boost::proto::value(expr)))
    {
    }

    template<class Xpr>
    void reset(Xpr const& other)
    {
      view tmp(other);
      boost::proto::value(*this) = boost::proto::value(tmp);
      this->size_ = tmp.size_;
    }

    //==========================================================================
    // Enable base expression handling of assignment
    //==========================================================================
    template<class Xpr> BOOST_FORCEINLINE
    typename boost::disable_if< boost::is_base_of<nt2_expression, Xpr>
                              , view&
                              >::type
    operator=(Xpr const& xpr)
    {
      nt2_expression::operator=(xpr);
      return *this;
    }

    BOOST_FORCEINLINE view& operator=(view const& xpr)
    {
      nt2_expression::operator=(xpr);
      return *this;
    }

    template<class Xpr> BOOST_FORCEINLINE
    typename boost::disable_if< boost::is_base_of<nt2_expression, Xpr>
                              , view const&
                              >::type
    operator=(Xpr const& xpr) const
    {
      nt2_expression::operator=(xpr);
      return *this;
    }

    BOOST_FORCEINLINE view const& operator=(view const& xpr) const
    {
      nt2_expression::operator=(xpr);
      return *this;
    }
  };
} }

namespace nt2 { using nt2::container::view; }

#endif
