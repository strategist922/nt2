//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_FUNCTIONS_SIZE_HPP_INCLUDED
#define NT2_CORE_FUNCTIONS_SIZE_HPP_INCLUDED

#include <nt2/include/functor.hpp>

namespace nt2
{
  namespace tag
  {
    struct size_ : ext::abstract_<size_>
    {
      typedef ext::abstract_<size_> parent;
      template<class... Args>
      static BOOST_FORCEINLINE BOOST_AUTO_DECLTYPE dispatch(Args&&... args)
      BOOST_AUTO_DECLTYPE_BODY( dispatching_size_( ext::adl_helper(), static_cast<Args&&>(args)... ) )
    };
  }
  namespace ext
  {
    template<class Site, class... Ts>
    BOOST_FORCEINLINE generic_dispatcher<tag::size_, Site> dispatching_size_(adl_helper, boost::dispatch::meta::unknown_<Site>, boost::dispatch::meta::unknown_<Ts>...)
    {
      return generic_dispatcher<tag::size_, Site>();
    }
    template<class... Args>
    struct impl_size_;
  }

  NT2_FUNCTION_IMPLEMENTATION(nt2::tag::size_, size, 1)
  NT2_FUNCTION_IMPLEMENTATION(nt2::tag::size_, size, 2)
}

#endif
