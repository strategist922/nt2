//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_COMPUTE_COST_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_COMPUTE_COST_HPP_INCLUDED

#include <nt2/sdk/meta/is_container.hpp>
#include <nt2/sdk/shared_memory/details/aggregate_costs.hpp>

#include <algorithm>
#include <set>

#include <boost/proto/proto.hpp>
#include <boost/mpl/int.hpp>


namespace nt2 { namespace details
{
    template < class In, class Out >
    inline bool compute_cost(In & in, Out & out)
    {
      typedef typename In::const_pointer raw_type;

      std::set< raw_type > terminal_set;
      details::aggregate_terminals()(in,  0, terminal_set);
      details::aggregate_terminals()(out, 0, terminal_set);

      return terminal_set.size() < details::aggregate_costs()(in);
    }

} }

#endif
