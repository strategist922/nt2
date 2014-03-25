//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_SPAWNER_TRANSFORM_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_SPAWNER_TRANSFORM_HPP_INCLUDED

#include <nt2/sdk/shared_memory.hpp>
#include <nt2/sdk/shared_memory/spawner.hpp>
#include <nt2/sdk/shared_memory/future.hpp>

#include <nt2/sdk/shared_memory/details/then_worker.hpp>
#include <nt2/sdk/shared_memory/details/aggregate_nodes.hpp>
#include <nt2/sdk/shared_memory/details/proto_data_with_futures.hpp>
#include <nt2/sdk/shared_memory/details/insert_dependencies.hpp>

#include <nt2/sdk/shared_memory/settings/container_has_futures.hpp>

#include <pair>

#ifndef BOOST_NO_EXCEPTIONS
#include <boost/exception_ptr.hpp>
#endif


namespace nt2
{
    namespace tag
    {
        struct transform_;
        template<class Arch> struct asynchronous_;
    }

    template<class Arch>
    struct spawner< tag::transform_, tag::asynchronous_<Arch> >
    {
        spawner(){}

        template<typename Worker,typename Pair>
        void operator()(Worker & w
                       ,Pair begin
                       ,Pair size
                       ,Pair grain_out
                       )
        {
             typedef typename
             nt2::make_future< Arch ,int >::type future;

             typedef typename
             details::container_has_futures<Arch>::call_it call_it;

             typedef typename
             std::vector<future>::iterator future_it;

             std::size_t condition_row = size.first / grain_out.first;
             std::size_t condition_col = size.second / grain_out.second;

             std::size_t leftover_row = size.first % grain_out.first;
             std::size_t leftover_col = size.second % grain_out.second;

             std::size_t nblocks_row  = condition_row ? condition_row : 1;
             std::size_t nblocks_col  = condition_col ? condition_col : 1;

             std::size_t last_chunk_row = condition_row ? grain_out.first  + leftover_row : size.first;
             std::size_t last_chunk_col = condition_col ? grain_out.second + leftover_col : size.second;

             details::container_has_futures<Arch> * pout_specifics;
             details::aggregate_specifics()(w.out_, 0, pout_specifics);
             details::container_has_futures<Arch> & out_specifics = * pout_specifics;

             details::container_has_futures<Arch> tmp;

             tmp.grain_ = std::make_pair(
                            condition_row ? grain_out.first  : size.first
                           ,condition_col ? grain_out.second : size.second
                           );

             tmp.futures_.reserve(nblocks_row*nblocks_col);

             details::aggregate_futures aggregate_f;

             #ifndef BOOST_NO_EXCEPTIONS
             boost::exception_ptr exception;

             try
             {
             #endif


             for(std::size_t nn=0, n=begin.first; nn<nblocks_row; ++nn, n+=grain_out.first)
             {
                for(std::size_t mm=0, m=begin.second; mm<nblocks_col; ++mm, m+=grain_out.second)
                {
                     std::size_t chunk_n = (nn<nblocks_row-1) ? grain_out.first  : last_chunk_row;
                     std::size_t chunk_m = (mm<nblocks_col-1) ? grain_out.second : last_chunk_col;

                     std::pair<std::size_t,std::size_t> offset (n,m);
                     std::pair<std::size_t,std::size_t> chunk (chunk_n,chunk_m);

                     details::proto_data_with_futures< future
                      ,details::container_has_futures<Arch>
                      > data_in ( offset, chunk, out_specifics );

                    for(call_it i=out_specifics.calling_cards_.begin();
                         i!=out_specifics.calling_cards_.end();
                         ++i)
                     {
                        details::insert_dependencies(
                            data_in.futures_, offset , chunk, grain_out
                           ,(*i)->futures_ , (*i)->grain_
                          );
                     }

                     aggregate_f(w.in_,0,data_in);

                     if(data_in.futures_.empty())
                     {
                         // Call operation
                         tmp.futures_.push_back(
                           nt2::async<Arch>(Worker(w), offset, chunk)
                             );
                     }

                     else
                     {
                         // Call operation
                         tmp.futures_.push_back(
                            nt2::when_all<Arch>(boost::move(data_in.futures_))
                            .then( details::then_worker<Worker>(Worker(w),offset, chunk)
                              )
                            );
                     }
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
        }
    };
}

#endif
