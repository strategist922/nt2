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
                                     std::size_t begin,
                                     std::size_t grain_out,
                                     FutureVector & in,
                                     std::size_t grain_in
                                    )
    {
        typedef typename FutureVector::iterator Iterator;

        Iterator begin_dep  = in.begin() + begin/grain_in;

        Iterator end_dep    = ( (begin + grain_out) % grain_in )
        ? in.begin() + std::min( in.size(), (begin + grain_out)/grain_in + 1)
        : in.begin() + (begin + grain_out)/grain_in;

        // Push back the dependencies
        out.insert(out.end(),begin_dep,end_dep);
    }

} }

#endif
