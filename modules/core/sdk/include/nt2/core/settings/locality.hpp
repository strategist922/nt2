//==============================================================================
//         Copyright 2009 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_SETTINGS_LOCALITY_HPP_INCLUDED
#define NT2_CORE_SETTINGS_LOCALITY_HPP_INCLUDED

#include <nt2/core/settings/forward/locality.hpp>

namespace nt2 { namespace tag
{
  struct locality_
  {
    template<class T>
    struct apply : boost::mpl::false_
    {};

    typedef nt2::host_ default_type;
  };

  template<>
  struct locality_::apply<nt2::host_>
                        : boost::mpl::true_
  {};

  template<>
  struct locality_::apply<nt2::device_>
                        : boost::mpl::true_
  {};
} }

#include <nt2/core/settings/details/locality.hpp>

#endif
