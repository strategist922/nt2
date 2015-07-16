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
#include <nt2/sdk/bench/metric/absolute_cycles.hpp>
#include <nt2/sdk/bench/setup/fixed.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

#include <nt2/table.hpp>


using namespace nt2::bench;
using boost::dispatch::default_site;
using nt2::table;

//==============================================================================
// scan spawner microbenchmark
//==============================================================================
struct shared_memory_scan
{
  shared_memory_scan(std::size_t n)
  :  n_(n),w_(out_,in_)
  {
    offset_ = w_.setdelaylength(0.1e-6) * n_ / nt2::get_num_threads();
  }

  float operator()() {
     return s_(w_, 0, n_, 1);
   }

  friend std::ostream& operator<<(std::ostream& os, shared_memory_scan const& p)
  {
    return os << "(" << p.n_ << ")";
  }

  nt2::cycles_t offset() const { return offset_; }
  std::size_t size() const { return n_; }

  private:

  nt2::table<double> out_, in_;
  std::size_t n_;
  nt2::spawner< nt2::tag::scan_
              , boost::dispatch::default_site<void>::type
              , float
              > s_;
  nt2::worker< nt2::tag::delay_
             ,void
             ,void
             ,nt2::table<double>
             ,nt2::table<double>
             > w_;
  nt2::cycles_t offset_;
};


NT2_REGISTER_BENCHMARK( shared_memory_scan )
{
  run_during_with< shared_memory_scan >( 1.
                                , fixed_<std::size_t>(10)
                                , absolute_cycles<stats::median_>()
                                );
}
