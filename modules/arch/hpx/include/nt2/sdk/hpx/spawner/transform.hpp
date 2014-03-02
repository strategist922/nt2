//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_HPX_SPAWNER_TRANSFORM_HPP_INCLUDED
#define NT2_SDK_HPX_SPAWNER_TRANSFORM_HPP_INCLUDED

#if defined(NT2_USE_HPX)

#include <nt2/sdk/shared_memory.hpp>
#include <nt2/sdk/shared_memory/spawner.hpp>
#include <nt2/sdk/hpx/settings/specific_data.hpp>
#include <nt2/sdk/shared_memory/details/aggregate_nodes.hpp>

#include <nt2/sdk/hpx/future/future.hpp>
#include <nt2/sdk/hpx/future/when_all.hpp>
#include <hpx/lcos/wait_all.hpp>

#include <nt2/sdk/shared_memory/details/then_worker.hpp>
#include <nt2/sdk/shared_memory/details/proto_data_with_futures.hpp>
#include <nt2/sdk/shared_memory/settings/container_has_futures.hpp>

#include <vector>

#ifndef BOOST_NO_EXCEPTIONS
#include <boost/exception_ptr.hpp>
#endif


namespace nt2
{
    namespace tag
    {
        struct transform_;
    }

    template<class Site>
    struct spawner< tag::transform_, tag::hpx_<Site> >
    {

        spawner() {}

        template<typename Worker>
        void operator()(Worker & w, std::size_t begin, std::size_t size, std::size_t grain_out)
        {
            typedef typename tag::hpx_<Site> Arch;

            typedef typename
            nt2::make_future< Arch ,int >::type future;

            typedef typename
            details::container_has_futures<Arch>::call_it call_it;

            typedef typename
            std::vector<future>::iterator future_it;

            std::size_t condition = size/grain_out;
            std::size_t leftover = size % grain_out;

            std::size_t nblocks  = condition ? condition : 1;
            std::size_t last_chunk = condition ? grain_out+leftover : size;

            std::size_t grain_in;

//            details::container_has_futures<Arch> * pout_specifics;
//            details::aggregate_specifics()(w.out_, 0, pout_specifics);
//            details::container_has_futures<Arch> & out_specifics = *pout_specifics;

            details::container_has_futures<Arch> &
            out_specifics( boost::proto::value(w.out_).specifics() );

            details::container_has_futures<Arch> tmp;

            tmp.grain_ = condition ? grain_out : size;
            tmp.futures_.reserve(nblocks);

            details::aggregate_futures aggregate_f;

#ifndef BOOST_NO_EXCEPTIONS
            boost::exception_ptr exception;

            try
            {
#endif

                for(std::size_t n=0, offset=begin; n<nblocks; ++n, offset+=grain_out)
                {
                    std::size_t chunk = (n<nblocks-1) ? grain_out : last_chunk;

                    details::proto_data_with_futures< future
                    ,details::container_has_futures<Arch>
                    > data_in(offset,chunk,out_specifics);

//                for(call_it i=out_specifics.calling_cards_.begin();
//                     i!=out_specifics.calling_cards_.end();
//                     ++i)
//                 {
//                     std::size_t grain_in = (*i)->grain_;
//
//                     future_it begin_dep = (*i)->futures_.begin() + offset/grain_in;
//
//                     future_it end_dep   = ( (offset + chunk) % grain_in )
//                     ? (*i)->futures_.begin() +
//                     std::min( (*i)->futures_.size(), (offset + chunk)/grain_in + 1)
//                     : (*i)->futures_.begin() + (offset + chunk)/grain_in;
//
//                     // Push back the dependencies
//                     data_in.futures_.insert(data_in.futures_.end(),begin_dep,end_dep);
//                 }

                    aggregate_f(w.in_,0,data_in);

                    if(data_in.futures_.empty())
                    {
                        // Call operation
                        tmp.futures_.push_back (
                            async<Arch>(Worker(w), offset, chunk)
                            );
                    }

                    else
                    {
                        // Call operation
                        tmp.futures_.push_back(
                           when_all<Arch>(boost::move(data_in.futures_))
                           .then(details::then_worker<Worker>
                                 (Worker(w),offset, chunk)
                                 )
                           );
                    }
                }

                out_specifics.swap(tmp);

#ifndef BOOST_NO_EXCEPTIONS
            }
            catch(...)
            {
                exception = boost::current_exception();
            }
#endif


//           typedef typename
//           nt2::make_future< Arch,int >::type future;
//
//           std::size_t condition = size/grain_out;
//           std::size_t leftover = size % grain_out;
//
//           std::size_t nblocks  = condition ? condition : 1;
//           std::size_t last_chunk = condition ? grain_out+leftover : size;
//
//           std::vector< future > barrier;
//           barrier.reserve(nblocks);
//
//           #ifndef BOOST_NO_EXCEPTIONS
//           boost::exception_ptr exception;
//
//           try
//           {
//           #endif
//
//           for(std::size_t n=0;n<nblocks;++n)
//           {
//             std::size_t chunk = (n<nblocks-1) ? grain_out : last_chunk;
//             // Call operation
//             barrier.push_back ( async<Arch>(w, begin+n*grain_out, chunk) );
//           }
//
//           for(std::size_t n=0;n<nblocks;++n)
//           {
//               // Call operation
//               barrier[n].get();
//           }
//
//           #ifndef BOOST_NO_EXCEPTIONS
//           }
//           catch(...)
//           {
//               exception = boost::current_exception();
//           }
//           #endif

        }
    };
}

#endif
#endif
