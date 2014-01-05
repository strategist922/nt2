//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_HPX_FUTURE_DETAILS_HPX_FUTURE_HPP_INCLUDED
#define NT2_SDK_HPX_FUTURE_DETAILS_HPX_FUTURE_HPP_INCLUDED

#if defined(NT2_USE_HPX)

#include <hpx/include/lcos.hpp>
#include <hpx/include/util.hpp>

#include <boost/move/move.hpp>

namespace nt2
{

    template<typename result_type>
    class tbb_future
    {
        BOOST_COPYABLE_AND_MOVABLE(tbb_future)
        hpx::lcos::future<result_type> f_;

    public:
        tbb_future(){}

        // Compiler-generated copy constructor...

        tbb_future(BOOST_RV_REF(tbb_future) x)             // Move ctor
        : f_(boost::move(x.f_)) { }

        tbb_future& operator=(BOOST_RV_REF(tbb_future) x)  // Move assign
        {
            f_  = boost::move(x.f_);
            return *this;
        }

        tbb_future& operator=(BOOST_COPY_ASSIGN_REF(tbb_future) x) // Copy assign
        {
            f_  = x.f_;
            return *this;
        }


        bool is_ready() const
        {
            return f_.is_ready();
        }

        void wait()
        {
            return f_.wait();
        }

        result_type get()
        {
            return f_.get();
        }

        // Overload of then method
        template<typename F>
        hpx_future<typename boost::result_of<F>::type>
        then(F& f)
        {
            return tbb_future(
              hpx::lcos::local::dataflow( \
                hpx::util::unwrapped(f),f_) \
              );
        }
    };
}

 #endif
#endif
