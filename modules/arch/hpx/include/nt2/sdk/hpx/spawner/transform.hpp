//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_HPX_SPAWNER_TRANSFORM_HPP_INCLUDED
#define NT2_SDK_HPX_SPAWNER_TRANSFORM_HPP_INCLUDED

#if defined(NT2_USE_HPX)

#include <nt2/sdk/shared_memory/spawner.hpp>
#include <nt2/sdk/shared_memory/future.hpp>
#include <nt2/sdk/shared_memory/details/then_worker.hpp>

#include <nt2/sdk/hpx/future/future.hpp>
#include <nt2/sdk/hpx/future/when_all.hpp>

#include <vector>

#ifndef BOOST_NO_EXCEPTIONS
#include <boost/exception_ptr.hpp>
#endif

namespace nt2
{
    namespace tag
    {
        struct transform_;
        template<class T> struct hpx_;
    }



    template<class Site>
    struct spawner< tag::transform_, tag::hpx_<Site> >
    {

        typedef typename tag::hpx_<Site> Arch;

        spawner() {}

        template<typename Worker>
        void operator()(Worker & w, std::size_t begin, std::size_t size, std::size_t grain)
        {
            typedef typename
            nt2::make_future< Arch,int >::type future;

            std::vector<future> & futures_in (w.in_.specifics().futures_);
            std::vector<future> & futures_out (w.out_.specifics().futures_);

            std::size_t leftover = size % grain;
            std::size_t nblocks  = size/grain;
            futures_out.reserve(nblocks);

            #ifndef BOOST_NO_EXCEPTIONS
            boost::exception_ptr exception;

            try
            {
            #endif

            if(futures_in.empty())
            {

                for(std::size_t n=0;n<nblocks;++n)
                {
                  std::size_t chunk = (n<nblocks-1) ? grain : grain+leftover;

                  // Call operation
                  futures_out.push_back ( async<Arch>(w, begin+n*grain, chunk) );
                }
            }

            else
            {
                nblocks = futures_in.size();

                for(std::size_t n=0;n<nblocks;++n)
                {
                  std::size_t chunk = (n<nblocks-1) ? grain : grain+leftover;

                  // Call operation
                  futures_out.push_back(
                    futures_in[n].then(
                        details::then_worker<Worker,Arch>
                        (w,begin+n*grain, chunk)
                    )
                  );
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
#endif
