//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_INSERT_DEPENDENCIES_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_INSERT_DEPENDENCIES_HPP_INCLUDED

#include <nt2/sdk/shared_memory/future.hpp>
#include <vector>

namespace nt2 { namespace details {

    template<typename FutureVector>
    inline void insert_dependencies( FutureVector & out,
                                     std::pair<std::size_t,std::size_t> begin,    // Indexes of the element in left up corner of Out tile
                                     std::pair<std::size_t,std::size_t> chunk,    // height/width of Out tile
                                     FutureVector & in,                           // vector of Futures In
                                     std::pair<std::size_t,std::size_t> grain_in, // Default height/width of In tile
                                     std::pair<std::size_t,std::size_t> LDX       // Height of In in number of In tiles
                                    )
    {

        std::size_t begin_m  = begin.first / grain_in.first;
        std::size_t end_m    = ( (begin.first + chunk.first) % grain_in.first )
        ? (begin.first + chunk.first) / grain_in.first + 1
        : (begin.first + chunk.first) / grain_in.first;

        end_m = std::min( LDX.first, end_m);


        std::size_t begin_n  = begin.second / grain_in.second;
        std::size_t end_n  = ( (begin.second + chunk.second) % grain_in.second )
        ? (begin.second + chunk.second) / grain_in.second + 1
        : (begin.second + chunk.second) / grain_in.second;

        end_n = std::min( LDX.second, end_n);

        for(std::size_t n = begin_n; n!= end_n; n++)
        {
            for(std::size_t m = begin_m; m!= end_m; m++)
            {
               out.push_back( in[m+n*LDX.first] );
            }
        }

    }

} }

#endif
