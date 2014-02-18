//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2011   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_SETTINGS_CONTAINER_HAS_FUTURES_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_SETTINGS_CONTAINER_HAS_FUTURES_HPP_INCLUDED

#include <nt2/sdk/shared_memory/future.hpp>
#include <boost/swap.hpp>
#include <vector>


namespace nt2 { namespace details {

  template <typename Arch>
  struct container_has_futures
  {
    typedef typename nt2::make_future<Arch,int >::type future;

    inline void swap(container_has_futures& src)
    {
      boost::swap(futures_,src.futures_);
      boost::swap(grain_,src.grain_);

      // Clear previous futures to avoid premature
      // synchronization
      src.futures_.clear();
    }

    inline void synchronize()
    {
        for(std::size_t n=0;n<futures_.size();++n)
        {
            futures_[n].get();
        }
        futures_.clear();

        for(std::size_t n=0;n<calling_cards_.size();++n)
        {
            calling_cards_[n]->synchronize();
        }
        calling_cards_.clear();
    }

    ~container_has_futures()
    {
      if (!futures_.empty()) synchronize();
    }

    //===========================================
    // vector of Futures
    //===========================================
    std::vector<future> futures_;
    std::vector< container_has_futures *> calling_cards_;
    std::size_t grain_;

  };

} }

#endif
