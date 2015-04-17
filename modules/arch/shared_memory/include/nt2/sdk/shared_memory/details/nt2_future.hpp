//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_DETAILS_NT2_FUTURE_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_DETAILS_NT2_FUTURE_HPP_INCLUDED

#include <type_traits>
#include <future>

namespace nt2
{
  namespace details
  {
    template<typename result_type>
    struct nt2_future
    : public std::future<result_type>
    {
      nt2_future()
      : std::future<result_type>()
      {}

      nt2_future( std::future<result_type> && other)
      : std::future<result_type>(
        std::forward< std::future<result_type> >(other)
        )
      {}

      template<typename F>
      details::nt2_future<
      typename std::result_of<F(nt2_future)>::type
      >
      then(F&& f)
      {
        auto packaged_task = [this,f]()
                              {
                                return f( std::move(*this) );
                              };

        return std::async(packaged_task);
      }

    };
  }
}

#endif
