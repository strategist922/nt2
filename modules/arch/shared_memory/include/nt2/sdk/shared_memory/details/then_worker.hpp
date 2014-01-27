//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_THEN_WORKER_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_THEN_WORKER_HPP_INCLUDED

#include <nt2/sdk/shared_memory/future.hpp>

namespace nt2 { namespace details {

    template<class Worker,class Arch>
    struct then_worker
    {
        typedef int result_type;

        typedef typename
        nt2::make_future< Arch,int >::type future;

        then_worker(Worker & w,
                    std::size_t begin,
                    std::size_t size
                    )
        :w_(w),begin_(begin)
        {}

        int operator()(future) const
        {
            w(begin,size);
        }

        Worker & w_;
        std::size_t begin_;
        std::size_t size_;

        private:
        then_worker& operator=(then_worker const&);
    };

} }

#endif
