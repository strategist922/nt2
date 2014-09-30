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
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/setup/fixed.hpp>
#include <nt2/sdk/bench/setup/combination.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

#include <boost/mpl/integral_c.hpp>

#include <nt2/table.hpp>


using namespace nt2::bench;
using boost::dispatch::default_site;
using nt2::table;

//==============================================================================
// transform spawner microbenchmark
//==============================================================================
struct shared_memory_transform
{
  shared_memory_transform(std::size_t n)
  :  n_(n),w_(out_,in_)
  {
    nt2::set_num_threads(n);
    w_.setdelaylength(0.1e-6);
  }

  void operator()() {
     s_(w_, 0, n_, 1);
   }

  friend std::ostream& operator<<(std::ostream& os, shared_memory_transform const& p)
  {
    return os << "(" << p.n_ << ")";
  }

  std::size_t size() const { return 1; }

  private:

  nt2::table<double> out_, in_;
  std::size_t n_;
  nt2::spawner< nt2::tag::transform_
              , boost::dispatch::default_site<void>::type
              > s_;
  nt2::worker< nt2::tag::delay_
             ,void
             ,void
             ,nt2::table<double>
             ,nt2::table<double>
             > w_;
};


NT2_REGISTER_BENCHMARK( shared_memory_transform )
{
  std::size_t max_threads = nt2::get_num_threads();

  run_during_with< shared_memory_transform >( 1.
                                  , fixed_<std::size_t>(max_threads)
                                  , cycles_per_element<stats::median_>()
                                  );
}
