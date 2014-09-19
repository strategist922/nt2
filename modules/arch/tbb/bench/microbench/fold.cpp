//==============================================================================
//         Copyright 2009 - 2013   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013   MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/sdk/shared_memory/spawner.hpp>
#include <nt2/sdk/shared_memory/worker/delay.hpp>
#include <nt2/sdk/shared_memory/thread_utility.hpp>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/absolute_time.hpp>
#include <nt2/sdk/bench/setup/arithmetic.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

#include <nt2/table.hpp>


using namespace nt2::bench;
using boost::dispatch::default_site;
using nt2::table;

//==============================================================================
// fold spawner microbenchmark
//==============================================================================
struct shared_memory_fold
{
  shared_memory_fold(std::size_t n)
  :  n_(n),w_(out_,in_)
  {
    nt2::set_num_threads(n);
    w_.setdelaylength(0.1e-6);
  }

  float operator()() {
     return s_(w_, 0, n_, 1);
   }

  friend std::ostream& operator<<(std::ostream& os, shared_memory_fold const& p)
  {
    return os << "(" << p.n_ << ")";
  }

  std::size_t size() const { return 1; }

  private:

  nt2::table<double> out_, in_;
  std::size_t n_;
  nt2::spawner< nt2::tag::fold_
              , boost::dispatch::default_site<void>::type
              , float
              > s_;
  nt2::worker< nt2::tag::delay_
             ,void
             ,void
             ,nt2::table<double>
             ,nt2::table<double>
             > w_;
};


NT2_REGISTER_BENCHMARK( shared_memory_fold )
{
  std::size_t max_threads = nt2::get_num_threads();

  run_during_with< shared_memory_fold >( 1.
                                  , arithmetic(1,max_threads,1)
                                  , absolute_time<stats::median_>()
                                  );
}
