//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_TBB_TBB_INITIALIZER_HPP_INCLUDED
#define NT2_SDK_TBB_TBB_INITIALIZER_HPP_INCLUDED

#include <tbb/task_scheduler_init.h>

namespace nt2 {

class tbb_initializer{
public:
    static void init(int n)
    {
       kill();
       init_ = new tbb::task_scheduler_init(n);
    }

    static void kill()
    {
        if (NULL != init_)
        {
           delete init_;
        }
    }

private:
static tbb::task_scheduler_init * init_;
};

tbb::task_scheduler_init *
tbb_initializer::init_ = NULL;

} // namespace nt2

#endif
