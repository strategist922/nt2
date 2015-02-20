//==============================================================================
//         Copyright 2014 - 2015   LRI    UMR 8623 CNRS/Univ Paris Sud XI
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <nt2/table.hpp>
#include <nt2/include/functions/ones.hpp>
#include <nt2/include/functions/copy.hpp>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/gb_per_second.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/setup/geometric.hpp>
#include <nt2/sdk/bench/setup/constant.hpp>
#include <nt2/sdk/bench/setup/combination.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

using namespace nt2::bench;
using namespace nt2;

template<typename T>
struct cuda_nt2
{
  public:
  cuda_nt2( std::size_t s )
                  : size_(s)
  {
    X.resize(nt2::of_size(size_));
    for(std::size_t i = 1; i<=size_; ++i)
      X(i) = T(i-1);
  }

  void operator()()
  {
    Y = X;
    X = Y;
  }

  std::size_t size() const { return size_*sizeof(T); }

  private:
  std::size_t size_;
  nt2::table<T> X;
  nt2::table<T,nt2::device_ > Y;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, cuda_nt2<T> const& p)
{
  return os << "(" << p.size() << ")";
}

NT2_REGISTER_BENCHMARK_TPL( cuda_nt2, (double) )
{
  std::size_t size_min  = args("size_min",   100*100);
  std::size_t size_max  = args("size_max", 15000*15000);
  std::size_t size_step = args("size_step",   20);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std:: cout << "Device Name: " <<  prop.name << std::endl;
  std:: cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;

  run_during_with< cuda_nt2<T> > ( 4.
                                 , geometric(size_min,size_max,size_step)
                                 , gb_per_second<stats::median_>(2)
                                 );
}
