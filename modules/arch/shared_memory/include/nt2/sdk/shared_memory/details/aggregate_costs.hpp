//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_AGGREGATE_COSTS_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_AGGREGATE_COSTS_HPP_INCLUDED

#include <nt2/sdk/meta/is_container.hpp>

#include <algorithm>

#include <boost/proto/proto.hpp>
#include <boost/mpl/size_t.hpp>

namespace nt2 { namespace details
{
    struct iplus : std::plus<std::size_t>, boost::proto::callable {};

    struct aggregate_costs
    :boost::proto::or_<
      // If the expression is a terminal
      // return 0
     boost::proto::when<
          boost::proto::terminal< boost::proto::_ >
         ,boost::mpl::size_t<0>()
      >
     // Else recall aggregate_costs and increments the (reduced) returned value
     ,boost::proto::otherwise<
        boost::proto::fold<
          boost::proto::_
         ,boost::mpl::size_t<1>()
         ,iplus(boost::proto::_state,aggregate_costs)
        >
      >
    >
    {};


    struct insert_raw : boost::proto::callable
    {
      typedef int result_type;

      template < class Container, class Set >
      inline int operator()(Container & in, Set & data) const
      {
        data.insert( in.raw() );
        return 0;
      }

    };

    struct aggregate_terminals
    :boost::proto::or_<
      // If the expression is a non-terminal
      // Do nothing
     boost::proto::when<
        boost::proto::and_<
          boost::proto::terminal< boost::proto::_ >
         ,boost::proto::not_<
           boost::proto::if_<
              meta::is_container_or_ref< boost::proto::_value>()
           >
          >
        >
     , boost::mpl::size_t<0>()
     >
      // Else if the expression is a container terminal
      // call insert_raw
     ,boost::proto::when<
        boost::proto::and_<
          boost::proto::terminal< boost::proto::_ >
         ,boost::proto::if_<
            meta::is_container_or_ref< boost::proto::_value>()
         >
        >
      ,insert_raw(boost::proto::_value,boost::proto::_data)
      >
     // Else recall aggregate_terminals with child nodes
     ,boost::proto::otherwise<
        boost::proto::fold<
          boost::proto::_
         ,boost::mpl::size_t<0>()
         ,aggregate_terminals
        >
      >
    >
    {};

} }

#endif
