//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_CORE_CONTAINER_DSL_DETAILS_SPECIFIC_DATA_HPP_INCLUDED
#define NT2_CORE_CONTAINER_DSL_DETAILS_SPECIFIC_DATA_HPP_INCLUDED

#include <nt2/core/settings/specific_data.hpp>
#include <nt2/dsl/functions/terminal.hpp>
#include <nt2/sdk/shared_memory.hpp>

namespace nt2 { namespace details {

  template<class Arch>
  struct terminal_specific_data
  {
  public:
    specific_data()
    {}

    specific_data(specific_data const& s)
    {}

    ~specific_data()
    {}

    inline void swap(specific_data& src)
    {
    }

    //===========================================
    //
    //===========================================
    inline void synchronize()
    {
    }

    //===========================================
    // vector of Futures
    //===========================================

  };

} }

namespace nt2
{
  template <typename T>
  struct specific_data<nt2::tag::terminal_, Arch >
  {
    typedef typename
    details::terminal_specific_data<Arch> type;
  };
}

#endif
