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
#include <nt2/sdk/shared_memory/settings/specific_data.hpp>

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
        void operator()(Worker & w, std::size_t begin, std::size_t size, std::size_t grain_out)
        {
            typedef typename
            nt2::make_future< Arch,int >::type future;

            std::size_t nblocks  = size/grain_out;
            std::size_t leftover = size % grain_out;
            std::size_t grain_in = boost::proto::value(w.in_).specifics().grain_;

            std::vector<future> & futures_in  ( boost::proto::value(w.in_).specifics().futures_ );
            std::vector<future> & futures_out ( boost::proto::value(w.out_).specifics().futures_ );
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
                  std::size_t chunk = (n<nblocks-1) ? grain_out : grain_out+leftover;

                  // Call operation
                  futures_out.push_back ( async<Arch>(w, begin+n*grain_out, chunk) );
                }
            }

            else
            {
                for(std::size_t n=0;n<nblocks;++n)
                {
                   std::size_t begin = grain_in*n/grain_out;
                   std::size_t end = grain_in*(n+1)/grain_out;
                   std::size_t chunk = (n<nblocks-1) ? grain_out : grain_out+leftover;

                   // Call operation
                   futures_out.push_back(
                     when_all<Arch>(&futures_in[begin],&futures_in[end])
                      .then(details::then_worker<Worker,Arch>
                        (w,begin+n*grain_out, chunk)
                       )
                  );
                }
            }

            boost::proto::value(w.out_).specifics().grain_ = grain_out;

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
