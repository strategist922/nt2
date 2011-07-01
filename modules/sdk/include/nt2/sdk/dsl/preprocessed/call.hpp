namespace nt2 { namespace meta
{
  template<class Func , class A0> inline typename boost::enable_if < any< boost::proto::is_expr<boost::mpl::_> , A0 > , nt2::meta::implement<Func(tag::ast_), tag::formal_ > >::type dispatching( Func const&, tag::formal_ const& , nt2::meta::unspecified_<A0> const& , adl_helper = adl_helper() ) { nt2::meta::implement<Func(tag::ast_), tag::formal_ > that; return that; } template<class Func , class A0 , class A1> inline typename boost::enable_if < any< boost::proto::is_expr<boost::mpl::_> , A0 , A1 > , nt2::meta::implement<Func(tag::ast_), tag::formal_ > >::type dispatching( Func const&, tag::formal_ const& , nt2::meta::unspecified_<A0> const& , nt2::meta::unspecified_<A1> const& , adl_helper = adl_helper() ) { nt2::meta::implement<Func(tag::ast_), tag::formal_ > that; return that; } template<class Func , class A0 , class A1 , class A2> inline typename boost::enable_if < any< boost::proto::is_expr<boost::mpl::_> , A0 , A1 , A2 > , nt2::meta::implement<Func(tag::ast_), tag::formal_ > >::type dispatching( Func const&, tag::formal_ const& , nt2::meta::unspecified_<A0> const& , nt2::meta::unspecified_<A1> const& , nt2::meta::unspecified_<A2> const& , adl_helper = adl_helper() ) { nt2::meta::implement<Func(tag::ast_), tag::formal_ > that; return that; } template<class Func , class A0 , class A1 , class A2 , class A3> inline typename boost::enable_if < any< boost::proto::is_expr<boost::mpl::_> , A0 , A1 , A2 , A3 > , nt2::meta::implement<Func(tag::ast_), tag::formal_ > >::type dispatching( Func const&, tag::formal_ const& , nt2::meta::unspecified_<A0> const& , nt2::meta::unspecified_<A1> const& , nt2::meta::unspecified_<A2> const& , nt2::meta::unspecified_<A3> const& , adl_helper = adl_helper() ) { nt2::meta::implement<Func(tag::ast_), tag::formal_ > that; return that; } template<class Func , class A0 , class A1 , class A2 , class A3 , class A4> inline typename boost::enable_if < any< boost::proto::is_expr<boost::mpl::_> , A0 , A1 , A2 , A3 , A4 > , nt2::meta::implement<Func(tag::ast_), tag::formal_ > >::type dispatching( Func const&, tag::formal_ const& , nt2::meta::unspecified_<A0> const& , nt2::meta::unspecified_<A1> const& , nt2::meta::unspecified_<A2> const& , nt2::meta::unspecified_<A3> const& , nt2::meta::unspecified_<A4> const& , adl_helper = adl_helper() ) { nt2::meta::implement<Func(tag::ast_), tag::formal_ > that; return that; }
  template<class Func,class Dummy>
  struct implement<Func(tag::ast_),tag::formal_,Dummy>
  {
    template<class Sig> struct result;
    template<class This,class A0> struct result<This(A0)> { typedef typename boost::proto::result_of:: make_expr < Func , typename nt2::details::result_of ::as_child< typename meta::strip< A0 >::type const& >::type >::type type; }; template<class This,class A0> inline typename result<implement (A0 const& ) >::type operator()(A0 const& a0 ) const { return boost::proto:: make_expr<Func>( nt2::meta::as_child(a0) ); } template<class This,class A0 , class A1> struct result<This(A0 , A1)> { typedef typename boost::proto::result_of:: make_expr < Func , typename nt2::details::result_of ::as_child< typename meta::strip< A0 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A1 >::type const& >::type >::type type; }; template<class This,class A0 , class A1> inline typename result<implement (A0 const& , A1 const& ) >::type operator()(A0 const& a0 , A1 const& a1 ) const { return boost::proto:: make_expr<Func>( nt2::meta::as_child(a0) , nt2::meta::as_child(a1) ); } template<class This,class A0 , class A1 , class A2> struct result<This(A0 , A1 , A2)> { typedef typename boost::proto::result_of:: make_expr < Func , typename nt2::details::result_of ::as_child< typename meta::strip< A0 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A1 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A2 >::type const& >::type >::type type; }; template<class This,class A0 , class A1 , class A2> inline typename result<implement (A0 const& , A1 const& , A2 const& ) >::type operator()(A0 const& a0 , A1 const& a1 , A2 const& a2 ) const { return boost::proto:: make_expr<Func>( nt2::meta::as_child(a0) , nt2::meta::as_child(a1) , nt2::meta::as_child(a2) ); } template<class This,class A0 , class A1 , class A2 , class A3> struct result<This(A0 , A1 , A2 , A3)> { typedef typename boost::proto::result_of:: make_expr < Func , typename nt2::details::result_of ::as_child< typename meta::strip< A0 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A1 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A2 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A3 >::type const& >::type >::type type; }; template<class This,class A0 , class A1 , class A2 , class A3> inline typename result<implement (A0 const& , A1 const& , A2 const& , A3 const& ) >::type operator()(A0 const& a0 , A1 const& a1 , A2 const& a2 , A3 const& a3 ) const { return boost::proto:: make_expr<Func>( nt2::meta::as_child(a0) , nt2::meta::as_child(a1) , nt2::meta::as_child(a2) , nt2::meta::as_child(a3) ); } template<class This,class A0 , class A1 , class A2 , class A3 , class A4> struct result<This(A0 , A1 , A2 , A3 , A4)> { typedef typename boost::proto::result_of:: make_expr < Func , typename nt2::details::result_of ::as_child< typename meta::strip< A0 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A1 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A2 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A3 >::type const& >::type , typename nt2::details::result_of ::as_child< typename meta::strip< A4 >::type const& >::type >::type type; }; template<class This,class A0 , class A1 , class A2 , class A3 , class A4> inline typename result<implement (A0 const& , A1 const& , A2 const& , A3 const& , A4 const& ) >::type operator()(A0 const& a0 , A1 const& a1 , A2 const& a2 , A3 const& a3 , A4 const& a4 ) const { return boost::proto:: make_expr<Func>( nt2::meta::as_child(a0) , nt2::meta::as_child(a1) , nt2::meta::as_child(a2) , nt2::meta::as_child(a3) , nt2::meta::as_child(a4) ); }
  };
} }
