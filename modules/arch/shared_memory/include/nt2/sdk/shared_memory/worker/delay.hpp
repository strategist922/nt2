//==============================================================================
//         Copyright 2003 - 2011   LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#ifndef NT2_SDK_SHARED_MEMORY_WORKER_DELAY_HPP_INCLUDED
#define NT2_SDK_SHARED_MEMORY_WORKER_DELAY_HPP_INCLUDED

#include <nt2/include/functions/zeros.hpp>
#include <nt2/sdk/shared_memory/worker.hpp>
#include <nt2/sdk/shared_memory/details/delay.hpp>
#include <nt2/sdk/timing/now.hpp>
#include <nt2/sdk/shared_memory/thread_utility.hpp>
#include <cstdio>


namespace nt2
{

  namespace tag
  {
    struct delay_;
  }

  // Transform Worker
  template<class Out, class In>
  struct worker<tag::delay_,void,void,Out,In>
  {
      worker(Out & out, In & in)
      :out_(out),in_(in),value_(100,0.)
      {}

      // Transform call operator
      void operator()(std::size_t rank, std::size_t)
      {
        nt2::details::delay(delaylength,value_[rank]);
      };

      // Fold call operator
      float operator()(float out, std::size_t rank, std::size_t)
      {
          float result = value_[rank] + out;
          nt2::details::delay(delaylength, result);
          return result;
      };

      // Scan call operator
      float operator()(float out, std::size_t rank, std::size_t, bool)
      {
          float result = value_[rank] + out;
          nt2::details::delay(delaylength, result);
          return result;
      };

      void setdelaylength(double delaytime) // in seconds
      {
          std::size_t reps = 1000;
          double lapsedtime = 0.0;
          double starttime;

          delaylength = 0;
          nt2::details::delay(delaylength, value_[0]);

          while (lapsedtime < delaytime)
          {
            delaylength = delaylength * 1.1 + 1;
            starttime = nt2::now();

            for (std::size_t i = 0; i < reps; i++)
              nt2::details::delay(delaylength,value_[0]);

            lapsedtime = (nt2::now() - starttime) / (double) reps;
          }
      }

      Out & out_;
      In & in_;
      std::plus<float> bop_;
      nt2::functor< nt2::tag::Zero > neutral_;
      std::size_t delaylength;
      std::vector<float> value_;

  private:
      worker& operator=(worker const&);
  };

}
#endif
