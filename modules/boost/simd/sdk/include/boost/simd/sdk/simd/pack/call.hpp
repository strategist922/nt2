/*******************************************************************************
 *         Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
 *         Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
 *
 *          Distributed under the Boost Software License, Version 1.0.
 *                 See accompanying file LICENSE.txt or copy at
 *                     http://www.boost.org/LICENSE_1_0.txt
 ******************************************************************************/
#ifndef BOOST_SIMD_SDK_SIMD_PACK_CALL_HPP_INCLUDED
#define BOOST_SIMD_SDK_SIMD_PACK_CALL_HPP_INCLUDED

namespace boost { namespace simd { namespace ext
{
  // default terminal
  BOOST_SIMD_FUNCTOR_IMPLEMENTATION( tag::terminal_,tag::cpu_
                            , (A0)
                            , (unspecified_<A0>)
                            )
  {
      
    template<class Sig>
    struct result;
    
    template<class This, class A0_>
    struct result<This(A0_)>
      : add_reference<A0_>
    {
    };

    template<class A0_>
    BOOST_DISPATCH_FORCE_INLINE A0_&
    operator()(A0_& a0) const
    {
      return a0;
    }
  };
  
  // need generic_ to avoid map specialization
  BOOST_SIMD_FUNCTOR_IMPLEMENTATION( tag::terminal_,tag::cpu_
                            , (A0)
                            , (generic_<unspecified_<A0> >)
                            )
  {
      
    template<class Sig>
    struct result;
    
    template<class This, class A0_>
    struct result<This(A0_)>
      : add_reference<A0_>
    {
    };

    template<class A0_>
    BOOST_DISPATCH_FORCE_INLINE A0_&
    operator()(A0_& a0) const
    {
      return a0;
    }
  };
  
  // workaround, constant functors need to be made less general
  BOOST_SIMD_FUNCTOR_IMPLEMENTATION( tag::terminal_, tag::cpu_, (A0)(X)
                            , ((target_< simd_< arithmetic_<A0>,X> >))
                            )
  {
    template<class Sig>
    struct result;
    
    template<class This, class A0_>
    struct result<This(A0_)>
      : add_reference<A0_>
    {
    };

    template<class A0_>
    BOOST_DISPATCH_FORCE_INLINE A0_&
    operator()(A0_& a0) const
    {
      return a0;
    }
  };
  
  // constants
  BOOST_SIMD_FUNCTOR_IMPLEMENTATION(Func, tag::formal_
                        , (Func)(A0)
                        , (target_< ast_<A0> >)
                        )
  {
    typedef typename proto::domain_of<typename A0::type>::type  domain;
    typedef dispatch::meta::
            as_<typename dispatch::meta::
                semantic_of<typename A0::type>::type
               >  value;
   
    typedef typename proto::result_of::
            make_expr<Func, domain, const value&>::type         result_type;
   
    BOOST_DISPATCH_FORCE_INLINE result_type
    operator()(A0 const& a0) const
    {
      return boost::proto::detail::
             make_expr_<Func, domain, const value&>()(value());
    }
  };

  // array case
  BOOST_SIMD_FUNCTOR_IMPLEMENTATION_TPL( tag::terminal_,tag::cpu_
                                , (class Value)(class State)
                                  (class Data)(std::size_t N)
                                , ((array_<scalar_< arithmetic_<Value > >,N>))
                                  ((target_<array_<scalar_< arithmetic_<State> >,N> >))
                                  (scalar_< integer_<Data> >)
                                )
{
    typedef typename Value::value_type result_type;

    inline result_type
    operator()( Value const& v, State const&, Data const& p ) const
    {
      return v[p];
    }
  };
} } }

#endif
