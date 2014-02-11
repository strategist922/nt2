//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_AGGREGATE_FUTURES_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_AGGREGATE_FUTURES_HPP_INCLUDED

#include <nt2/sdk/meta/is_container.hpp>
#include <nt2/sdk/shared_memory/future.hpp>
#include <vector>
#include <algorithm>

#include <boost/proto/proto.hpp>
#include <boost/mpl/int.hpp>

namespace nt2 { namespace details
{

    struct process_node : boost::proto::callable
    {
      typedef int result_type;

      template <class Container,class ProtoData>
      inline int operator()(Container & in, ProtoData & data) const
      {
        typedef typename ProtoData::FutureVector FutureVector;
        typedef typename FutureVector::iterator Iterator;

        FutureVector & futures_in = in.specifics().futures_;
        std::size_t grain_in = in.specifics().grain_;

        std::size_t begin(data.begin_);
        std::size_t size(data.size_);


        if (!futures_in.empty())
        {
          Iterator begin_dep = futures_in.begin() + begin/grain_in;
          Iterator end_dep   = ( (begin + size)%grain_in )
            ? futures_in.begin() +
                std::min(futures_in.size(),(begin +size)/grain_in + 1)
            : futures_in.begin() + (begin +size)/grain_in;

             // Call operation
          data.futures_.insert(data.futures_.end(),begin_dep,end_dep);
        }

        return 0;
      }

    };

    struct aggregate_futures
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
      // call display_success
     ,boost::proto::when<
        boost::proto::and_<
          boost::proto::terminal< boost::proto::_ >
         ,boost::proto::if_<
            meta::is_container_or_ref< boost::proto::_value>()
         >
        >
      ,process_node(boost::proto::_value,boost::proto::_data)
      >
     // Else recall aggregate_futures for every child
     ,boost::proto::otherwise<
        boost::proto::fold<
          boost::proto::_
         ,boost::proto::_state
         ,aggregate_futures
        >
      >
    >
    {};

} }

#endif
