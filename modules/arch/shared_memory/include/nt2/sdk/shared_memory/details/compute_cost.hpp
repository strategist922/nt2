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
#include <nt2/sdk/config/cache.hpp>
#include <nt2/sdk/shared_memory/details/aggregate_costs.hpp>
#include <nt2/sdk/shared_memory/thread_utility.hpp>
#include <nt2/sdk/shared_memory/runtime_costs.hpp>

#include <algorithm>
#include <set>

#include <boost/proto/proto.hpp>
#include <boost/mpl/int.hpp>


namespace nt2 { namespace details
{
    template < class Tag, class Arch, class Out, class In >
    inline bool compute_cost(Out & out, In & in)
    {
      typedef typename In::const_pointer raw_type;
      typedef typename nt2::runtime_costs<Tag,Arch>::type skel_cost_type;

      std::size_t nin  = nt2::numel(in);
      std::size_t nout = nt2::numel(out);

      std::size_t cache_size
        = config::top_cache_size(3)/sizeof(typename Out::value_type);

      std::set< raw_type > terminal_set;
      details::aggregate_terminals()(in,  0, terminal_set);

      std::size_t access_cost  = terminal_set.size() * nin + nout;
      std::size_t compute_cost = details::aggregate_costs()(in) * nin;
      std::size_t cache_cost = (access_cost/cache_size)*100;
      std::size_t skel_cost  = skel_cost_type();

      std::size_t sequential = access_cost + compute_cost + cache_cost;

      std::size_t parallel
        = (access_cost + compute_cost) / nt2::get_num_threads()
        + cache_cost + skel_cost;

      return parallel < sequential;
    }

} }

#endif
