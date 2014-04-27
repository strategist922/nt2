//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_AGGREGATE_NODES_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_AGGREGATE_NODES_HPP_INCLUDED

#include <nt2/sdk/meta/is_container.hpp>
#include <nt2/sdk/shared_memory/future.hpp>
#include <nt2/sdk/shared_memory/details/insert_dependencies.hpp>

#include <vector>
#include <algorithm>

#include <boost/proto/proto.hpp>
#include <boost/mpl/int.hpp>

namespace nt2 { namespace details
{
    struct synchronize_futures : boost::proto::callable
    {
      typedef int result_type;

      template <class Container,class ProtoData>
      inline int operator()(Container & in, ProtoData &) const
      {
        in.specifics().synchronize();
        return 0;
      }

    };

    struct get_futures : boost::proto::callable
    {
      typedef int result_type;

      template <class Container,class ProtoData>
      inline int operator()(Container & in, ProtoData & data) const
      {
        typedef typename ProtoData::FutureVector FutureVector;

        FutureVector & futures_in = in.specifics().futures_;

        if (!futures_in.empty())
        {
          details::insert_dependencies(
            data.futures_, data.begin_, data.chunk_,in.specifics()
            );

          // Leave the "calling card" of out
          in.specifics().calling_cards_.insert( &(data.specifics_) );
        }

        return 0;
      }

    };

    struct get_specifics : boost::proto::callable
    {
        typedef int result_type;

        template <class Container,class ProtoData>
        inline int operator()(Container & in, ProtoData & data) const
        {
            data = &in.specifics();
            return 0;
        }
    };

    template<class F>
    struct aggregate_nodes
    :boost::proto::or_<
      // If the expression is a non-container terminal
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
       ,boost::proto::_state
      >
      // Else if the expression is a container terminal
      // call F
     ,boost::proto::when<
        boost::proto::and_<
          boost::proto::terminal< boost::proto::_ >
         ,boost::proto::if_<
            meta::is_container_or_ref< boost::proto::_value>()
         >
        >
      ,F(boost::proto::_value,boost::proto::_data)
      >
     // Else recall fold_futures for every child
     ,boost::proto::otherwise<
        boost::proto::fold<
          boost::proto::_
         ,boost::proto::_state
         ,aggregate_nodes<F>
        >
      >
    >
    {};

    typedef aggregate_nodes<get_futures> aggregate_futures;
    typedef aggregate_nodes<get_specifics> aggregate_specifics;
    typedef aggregate_nodes<synchronize_futures> aggregate_and_synchronize;

} }

#endif
