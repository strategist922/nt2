//==============================================================================
//         Copyright 2003 & onward LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 & onward LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_GLOBALMAX_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_GLOBALMAX_HPP_INCLUDED

#include <nt2/include/functor.hpp>

namespace nt2
{
  namespace tag
  {
    /*!
      @brief Tag for the globalmax functor
    **/
     struct globalmax_ : ext::abstract_<globalmax_>
    {
      /// @brief Parent hierarchy
      typedef ext::abstract_<globalmax_> parent;
      template<class... Args>
      static BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE dispatch(Args&&... args)
      BOOST_AUTO_DECLTYPE_BODY( dispatching_globalmax_( ext::adl_helper(), static_cast<Args&&>(args)... ) )
    };
  }
  namespace ext
  {
    template<class Site, class... Ts>
    BOOST_FORCEINLINE generic_dispatcher<tag::globalmax_, Site> dispatching_globalmax_(adl_helper, boost::dispatch::meta::unknown_<Site>, boost::dispatch::meta::unknown_<Ts>...)
    {
      return generic_dispatcher<tag::globalmax_, Site>();
    }
    template<class... Args>
    struct impl_globalmax_;
  }

  /*!
    @brief maximum  of all the elements of a table expression and its position.

    Computes maximum of all the elements of a table expression and optionaly its linear index

    @par Semantic

    For any table expression @c t:

    @code
    T r = globalmax(t);
    @endcode

    is equivalent to:

    @code
    T r = max(a(_));
    @endcode

    and

    @code
    ptrdiff_t i;
    T m = globalmax(t, i);
    @endcode

    is equivalent to:

    @code
    T r = max(a(_));
    ptrdiff_t i =  globalfind(eq(a0, m))
    @endcode


    @see @funcref{colon}, @funcref{max}, @funcref{globalfind}
    @param a0 Table to process
    @param a1 optional L-value to receive the index

    @return An expression eventually evaluated to the result
  **/
  NT2_FUNCTION_IMPLEMENTATION_TPL(tag::globalmax_, globalmax,(A0 const&)(A1&),2)
  /// @overload
  NT2_FUNCTION_IMPLEMENTATION_TPL(tag::globalmax_, g_max ,(A0 const&)(A1&),2)
  /// @overload
  NT2_FUNCTION_IMPLEMENTATION(nt2::tag::globalmax_       , globalmax, 1)
  /// @overload
  NT2_FUNCTION_IMPLEMENTATION(nt2::tag::globalmax_       , g_max, 1)
}

#endif
