//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_PROTO_DATA_WITH_FUTURES_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_PROTO_DATA_WITH_FUTURES_HPP_INCLUDED

#include <nt2/sdk/shared_memory/future.hpp>
#include <vector>
#include <utility>

namespace nt2 { namespace details {

    template<class Future,class Specifics>
    struct proto_data_with_futures
    {
        typedef typename std::vector<Future> FutureVector;

        proto_data_with_futures(std::pair<std::size_t,std::size_t> begin
                               ,std::pair<std::size_t,std::size_t> size
                               ,std::size_t LDX
                               ,Specifics & specifics)
        :begin_(begin),size_(size),LDX_(LDX),specifics_(specifics)
        {}

        FutureVector futures_;
        std::pair<std::size_t,std::size_t> begin_;
        std::pair<std::size_t,std::size_t> size_;
        std::pair<std::size_t,std::size_t> LDX_;
        Specifics & specifics_;

        private:
        proto_data_with_futures& operator=(proto_data_with_futures const&);

    };

} }

#endif
