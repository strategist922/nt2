//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_WORKER_DELAY_WORKER_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_WORKER_DELAY_WORKER_HPP_INCLUDED

#include <nt2/sdk/shared_memory/worker.hpp>
#include <nt2/sdk/shared_memory/details/delay.hpp>
#include <nt2/sdk/timing/now.hpp>

namespace nt2
{

  namespace tag
  {
    struct delay_worker_;
  }

  // Transform Worker
  template<class BackEnd,class Site, class Out, class In>
  struct worker<tag::delay_worker_,BackEnd,Site,Out,In>
  {
      worker(Out & out, In & in, std::size_t delaylength)
      :out_(out),in_(in)
      {}

      void operator()(std::size_t, std::size_t)
      {
          for (std::size_t i = 0; i < delaylength; i++)
          nt2::details::delay(delaylength);
      };

      std::size_t setdelaylength(double delaytime = 0.1) // microseconds
      {
          std::size_t reps = 1000;
          double lapsedtime = 0.0, starttime; // seconds

          delaytime = delaytime/1.0E6;

          delaylength = 0;
          nt2::details::delay(delaylength);

          while (lapsedtime < delaytime)
          {
            delaylength = delaylength * 1.1 + 1;
            starttime = nt2::now();

            for (std::size_t i = 0; i < reps; i++)
            {
              nt2::details::delay(delaylength);
            }

            lapsedtime = (nt2::now() - starttime) / (double) reps;
          }
          return delaylength;
      }

      Out & out_;
      In & in_;
      std::size_t delaylength;

  private:
      worker& operator=(worker const&);
  };

}
#endif
