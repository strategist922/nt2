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
                                     std::pair<std::size_t,std::size_t> begin,
                                     std::pair<std::size_t,std::size_t> size,
                                     FutureVector & in,
                                     std::pair<std::size_t,std::size_t> grain_in,
                                     std::pair<std::size_t,std::size_t> LDX
                                    )
    {

        std::size_t begin_n  = begin.first / grain_in.first;
        std::size_t end_n    = ( (begin.first + size.first) % grain_in.first )
        ? (begin.first + size.first) / grain_in.first + 1
        : (begin.first + size.first) / grain_in.first;

        end_n = std::min( LDX.first, end_n);


        std::size_t begin_m  = begin.second / grain_in.second;
        std::size_t end_m  = ( (begin.second + size.second) % grain_in.second )
        ? (begin.second + size.second) / grain_in.second + 1
        : (begin.second + size.second) / grain_in.second;

        end_m = std::min( LDX.second, end_m);

        for(std::size_t n = begin_n; n!= end_n; n++)
        for(std::size_t m = begin_m; m!= end_m; m++)
        {
           out.push_back( in[n+m*LDX.first] );
        }

    }

} }

#endif
