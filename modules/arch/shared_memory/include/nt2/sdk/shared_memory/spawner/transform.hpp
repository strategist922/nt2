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

#include <utility>
#include <cstdio>

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

        template<typename Worker>
        void operator()(Worker & w, std::size_t offset, std::size_t size, std::pair<std::size_t,std::size_t> grain_out)
        {
             typedef typename
             nt2::make_future< Arch ,int >::type future;

             typedef typename
             details::container_has_futures<Arch>::call_it call_it;

             std::size_t bound  = w.bound_;
             details::container_has_futures<Arch> tmp;

             printf("bound: %lu size: %lu\n",bound,size);

        // 2D parameters of In table
             std::size_t height = (size <= bound) ? size : bound;
             std::size_t width  = (size <= bound) ? 1 : size/bound + (size%bound > 0);

             printf("Table to fold is: %lu * %lu\n",height,width);

        // Update grain_out to have a valid grain
              std::size_t condition_row = height / grain_out.first;
              std::size_t condition_col = width  / grain_out.second;

              tmp.grain_ = std::make_pair(
                condition_row ? grain_out.first : height
               ,condition_col ? grain_out.second : width
               );

        // A Tile has a surface of grain_out(1) x grain_out(2), check leftovers
             std::size_t leftover_row = height % tmp.grain_.first;
             std::size_t leftover_col = width  % tmp.grain_.second;

        // Height/Width of Out in number of tiles
             std::size_t nblocks_row  = height / tmp.grain_.first;;
             std::size_t nblocks_col  = width  / tmp.grain_.second;

             printf("Tile array is: %lu * %lu\n",nblocks_row,nblocks_col);

             std::size_t last_chunk_row = tmp.grain_.first  + leftover_row;
             std::size_t last_chunk_col = tmp.grain_.second + leftover_col;

             details::container_has_futures<Arch> * pout_specifics;
             details::aggregate_specifics()(w.out_, 0, pout_specifics);
             details::container_has_futures<Arch> & out_specifics = * pout_specifics;


             printf("Chosen grain is: %lu * %lu\n",tmp.grain_.first,tmp.grain_.second);

             tmp.LDX_   = std::make_pair(nblocks_row,nblocks_col);

             tmp.futures_.reserve(nblocks_row*nblocks_col);

             details::aggregate_futures aggregate_f;

             #ifndef BOOST_NO_EXCEPTIONS
             boost::exception_ptr exception;

             try
             {
             #endif


             for(std::size_t nn=0, n=0; nn<nblocks_col; ++nn, n+=tmp.grain_.second)
             {
                 for(std::size_t mm=0, m=0; mm<nblocks_row; ++mm, m+=tmp.grain_.first)
                 {
                     std::size_t chunk_m = (mm<nblocks_row-1) ? tmp.grain_.first   : last_chunk_row;
                     std::size_t chunk_n = (nn<nblocks_col-1) ? tmp.grain_.second  : last_chunk_col;

                     std::pair<std::size_t,std::size_t> begin (m,n);
                     std::pair<std::size_t,std::size_t> chunk (chunk_m,chunk_n);

                     details::proto_data_with_futures< future
                      ,details::container_has_futures<Arch>
                      > data_in ( begin, chunk, tmp.LDX_, out_specifics );

                    for(call_it i=out_specifics.calling_cards_.begin();
                         i!=out_specifics.calling_cards_.end();
                         ++i)
                     {
                        details::insert_dependencies(
                            data_in.futures_, begin , chunk
                           ,(*i)->futures_ , (*i)->grain_, (*i)->LDX_
                          );
                     }

                     aggregate_f(w.in_,0,data_in);

                     if(data_in.futures_.empty())
                     {
                         // Call operation
                         tmp.futures_.push_back(
                           nt2::async<Arch>(Worker(w), begin, chunk, offset, size)
                             );
                     }

                     else
                     {
                        printf("Continuation is called\n");

                         // Call operation
                         tmp.futures_.push_back(
                            nt2::when_all<Arch>(boost::move(data_in.futures_))
                            .then( details::then_worker<Worker>(Worker(w),begin, chunk, offset, size)
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
