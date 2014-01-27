//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_SETTINGS_SPECIFIC_DATA_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_SETTINGS_SPECIFIC_DATA_HPP_INCLUDED

#include <nt2/core/settings/specific_data.hpp>
#include <nt2/sdk/shared_memory.hpp>

#include <boost/swap.hpp>

#include <vector>

namespace nt2 { namespace details {

  template<class Arch>
  struct container_has_futures
  {
    typedef typename nt2::make_future< Arch,int >::type future;

    inline void swap(container_has_futures& src)
    {
      boost::swap(futures_,src.futures_);
    }

    inline void synchronize()
    {
        for(std::size_t n=0;n<futures.size();++n)
        {
            futures_[n].get();
        }
    }

    //===========================================
    // vector of Futures
    //===========================================
    std::vector<future> futures_;

  };

} }

namespace nt2
{
  template <typename Arch,typename Site>
  struct specific_data<Site, Arch >
  {
    typedef typename
    details::container_has_futures<Arch> type;
  };
}

#endif
