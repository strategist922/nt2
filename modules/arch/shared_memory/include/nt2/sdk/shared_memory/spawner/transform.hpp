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
        void operator()(Worker & w, std::size_t, std::size_t, std::size_t grain)
        {
             typedef typename
             nt2::make_future< Arch ,int >::type future;

             typedef typename
             details::container_has_futures<Arch>::call_it call_it;

             details::container_has_futures<Arch> * ps;
             details::aggregate_specifics()(w.out_, 0, ps);
             details::container_has_futures<Arch> & s = * ps;

             details::aggregate_futures aggregate_f;

        // 2D parameters of In table
             std::size_t height = w.bound_;
             std::size_t width  = w.obound_;

        // Update specifics if it is empty
             if ( s.empty() )
             {
                 s.grain_ = std::make_pair(
                  (height > grain)  ? grain : height
                 ,(width  > grain)  ? grain : width
                 );

                 s.NTiles_ = std::make_pair(
                  height / s.grain_.first
                 ,width  / s.grain_.second
                 );

                 s.size_ = std::make_pair(height,width);
                 s.futures_.resize(s.NTiles_.first * s.NTiles_.second);
             }

             #ifndef BOOST_NO_EXCEPTIONS
             boost::exception_ptr exception;

             try
             {
             #endif

        // A Tile has a surface of grain x grain, check leftovers
             const std::size_t leftover_row = height % s.grain_.first;
             const std::size_t leftover_col = width  % s.grain_.second;

             const std::size_t last_chunk_row =  s.grain_.first  + leftover_row;
             const std::size_t last_chunk_col =  s.grain_.second + leftover_col;

        // Height/Width of Out in number of tiles
             const std::size_t nblocks_row = s.NTiles_.first;
             const std::size_t nblocks_col = s.NTiles_.second;

             for(std::size_t nn=0, n=0; nn<nblocks_col; ++nn, n+=s.grain_.second)
             {
                 for(std::size_t mm=0, m=0; mm<nblocks_row; ++mm, m+=s.grain_.first)
                 {
                     std::size_t chunk_m = (mm<nblocks_row-1)
                                         ? s.grain_.first
                                         : last_chunk_row;

                     std::size_t chunk_n = (nn<nblocks_col-1)
                                         ? s.grain_.second
                                         : last_chunk_col;

                     std::pair<std::size_t,std::size_t> begin (m,n);
                     std::pair<std::size_t,std::size_t> chunk (chunk_m,chunk_n);

                     details::proto_data_with_futures< future
                      ,details::container_has_futures<Arch>
                      > data_in ( begin, chunk, s );

                    for(call_it i = s.calling_cards_.begin();
                         i != s.calling_cards_.end();
                         ++i)
                     {
                        details::insert_dependencies( data_in.futures_, begin , chunk, **i );
                     }

                     aggregate_f(w.in_,0,data_in);


                     switch( data_in.futures_.size() )
                     {
                     case 0:
                       s.tile(mm,nn) = nt2::async<Arch>(Worker(w), begin, chunk);
                     break;

                     case 1:
                       s.tile(mm,nn)
                       = data_in.futures_[0]
                         .then( details::then_worker<Worker>(Worker(w), begin, chunk));
                     break;

                     default:
                        s.tile(mm,nn)
                        = nt2::when_all<Arch>(boost::move(data_in.futures_))
                          .then( details::then_worker<Worker>(Worker(w), begin, chunk)
                        );
                     break;
                     }
                 }
             }

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
