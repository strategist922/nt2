//==============================================================================
//         Copyright 2014          LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2014          LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2014          NumScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/table.hpp>
#include <iostream>

#include <nt2/include/functions/conv.hpp>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/setup/fixed.hpp>
#include <nt2/sdk/bench/stats/median.hpp>
#include <nt2/sdk/bench/args.hpp>

#include <nt2/include/functions/tie.hpp>

#include <nt2/include/functions/linspace.hpp>
#include <nt2/include/functions/ones.hpp>

#include <nt2/sdk/bench/setup/geometric.hpp>
#include <nt2/sdk/bench/setup/combination.hpp>
#include <boost/fusion/include/at.hpp>
#include <nt2/sdk/bench/setup/constant.hpp>

using namespace nt2::bench;
using namespace nt2;

#define size_filter  5
// VALID FILTER

template<typename T>
struct test_conv_valid
{
  typedef nt2::table<T,nt2::_2D> tab_t;

  typedef void experiment_is_immutable;

  template<typename Setup>
  test_conv_valid ( Setup const& s)
              : data_size(boost::fusion::at_c<0>(s))
              , size_filt(boost::fusion::at_c<1>(s))
  {
    data = nt2::linspace<T>(T(1),T(data_size),data_size);
    filt = nt2::ones(nt2::of_size(1,size_filt),nt2::meta::as_<T>());
    res.resize(data.extent());
  }

  void operator()()
  {
    res = nt2::conv ( data, filt, nt2::valid_ );
  }

  friend std::ostream& operator<<(std::ostream& os, test_conv_valid<T> const& p)
  {
    return os << "(" << p.size() << " @ " << p.size_filt << ")";
  }

  std::size_t size() const { return data_size; }

  private:
    std::size_t data_size;
    nt2::table<T> data;
    std::size_t size_filt;
    nt2::table<T> filt;
    nt2::table<T> res;
};




template<typename T>
struct test_conv_valid_static
{
  typedef nt2::table<T,nt2::_2D> tab_t;

  typedef void experiment_is_immutable;

  template<typename Setup>
  test_conv_valid_static ( Setup const& s)
              : data_size(s)
  {
    data = nt2::linspace<T>(T(1),T(data_size),data_size);
    filt = nt2::ones(nt2::of_size(1,size_filter),nt2::meta::as_<T>());
    res.resize(data.extent());
  }

  void operator()()
  {
    res = nt2::conv ( data, filt, nt2::valid_ );
  }

  friend std::ostream& operator<<(std::ostream& os, test_conv_valid_static<T> const& p)
  {
    return os << "(" << p.size() << " @ " << size_filter << ")";
  }

  std::size_t size() const { return data_size; }

  private:
    std::size_t data_size;
    nt2::table<T> data;
    std::size_t size_filt;
    nt2::table<T,nt2::of_size_<1,size_filter> > filt;
    nt2::table<T> res;
};


// SAME FILTER

template<typename T>
struct test_conv_same
{
  typedef nt2::table<T,nt2::_2D> tab_t;

  typedef void experiment_is_immutable;

  template<typename Setup>
  test_conv_same ( Setup const& s)
              : data_size(boost::fusion::at_c<0>(s))
              , size_filt(boost::fusion::at_c<1>(s))
  {
    data = nt2::linspace<T>(T(1),T(data_size),data_size);
    filt = nt2::ones(nt2::of_size(1,size_filt),nt2::meta::as_<T>());
    res.resize(data.extent());
  }

  void operator()()
  {
    res = nt2::conv ( data, filt, nt2::valid_ );
  }

  friend std::ostream& operator<<(std::ostream& os, test_conv_same<T> const& p)
  {
    return os << "(" << p.size() << " @ " << p.size_filt << ")";
  }

  std::size_t size() const { return data_size; }

  private:
    std::size_t data_size;
    nt2::table<T> data;
    std::size_t size_filt;
    nt2::table<T> filt;
    nt2::table<T> res;
};




template<typename T>
struct test_conv_same_static
{
  typedef nt2::table<T,nt2::_2D> tab_t;

  typedef void experiment_is_immutable;

  template<typename Setup>
  test_conv_same_static ( Setup const& s)
              : data_size(s)
  {
    data = nt2::linspace<T>(T(1),T(data_size),data_size);
    filt = nt2::ones(nt2::of_size(1,size_filter),nt2::meta::as_<T>());
    res.resize(data.extent());
  }

  void operator()()
  {
    res = nt2::conv ( data, filt, nt2::valid_ );
  }

  friend std::ostream& operator<<(std::ostream& os, test_conv_same_static<T> const& p)
  {
    return os << "(" << p.size() << " @ " << size_filter << ")";
  }

  std::size_t size() const { return data_size; }

  private:
    std::size_t data_size;
    nt2::table<T> data;
    std::size_t size_filt;
    nt2::table<T,nt2::of_size_<1,size_filter> > filt;
    nt2::table<T> res;
};

// FULL FILTER

template<typename T>
struct test_conv_full
{
  typedef nt2::table<T,nt2::_2D> tab_t;

  typedef void experiment_is_immutable;

  template<typename Setup>
  test_conv_full ( Setup const& s)
              : data_size(boost::fusion::at_c<0>(s))
              , size_filt(boost::fusion::at_c<1>(s))
  {
    data = nt2::linspace<T>(T(1),T(data_size),data_size);
    filt = nt2::ones(nt2::of_size(1,size_filt),nt2::meta::as_<T>());
    res.resize(data.extent());
  }

  void operator()()
  {
    res = nt2::conv ( data, filt, nt2::valid_ );
  }

  friend std::ostream& operator<<(std::ostream& os, test_conv_full<T> const& p)
  {
    return os << "(" << p.size() << " @ " << p.size_filt << ")";
  }

  std::size_t size() const { return data_size; }

  private:
    std::size_t data_size;
    nt2::table<T> data;
    std::size_t size_filt;
    nt2::table<T> filt;
    nt2::table<T> res;
};




template<typename T>
struct test_conv_full_static
{
  typedef nt2::table<T,nt2::_2D> tab_t;

  typedef void experiment_is_immutable;

  template<typename Setup>
  test_conv_full_static ( Setup const& s)
              : data_size(s)
  {
    data = nt2::linspace<T>(T(1),T(data_size),data_size);
    filt = nt2::ones(nt2::of_size(1,size_filter),nt2::meta::as_<T>());
    res.resize(data.extent());
  }

  void operator()()
  {
    res = nt2::conv ( data, filt, nt2::valid_ );
  }

  friend std::ostream& operator<<(std::ostream& os, test_conv_full_static<T> const& p)
  {
    return os << "(" << p.size() << " @ " << size_filter << ")";
  }

  std::size_t size() const { return data_size; }

  private:
    std::size_t data_size;
    nt2::table<T> data;
    std::size_t size_filt;
    nt2::table<T,nt2::of_size_<1,size_filter> > filt;
    nt2::table<T> res;
};



//CALLS

//VALID
NT2_REGISTER_BENCHMARK_TPL( test_conv_valid, NT2_SIMD_REAL_TYPES )
{
  std::size_t data_min= args("min",100);
  std::size_t data_max= args("max",4096);
  std::size_t data_step= args("step",2);
  std::size_t filt_size= args("filter",size_filter);

  run_during_with< test_conv_valid<T> > ( 1.
                                    , and_( geometric(data_min,data_max,data_step)
                                          , constant(filt_size)
                                          )
                                    , cycles_per_element<stats::median_>()
                                    );
}

NT2_REGISTER_BENCHMARK_TPL( test_conv_valid_static, NT2_SIMD_REAL_TYPES )
{
  std::size_t data_min= args("min",100);
  std::size_t data_max= args("max",4096);
  std::size_t data_step= args("step",2);

  run_during_with< test_conv_valid_static<T> > ( 1.
                                    , geometric(data_min,data_max,data_step)
                                    , cycles_per_element<stats::median_>()
                                    );
}

//SAME

NT2_REGISTER_BENCHMARK_TPL( test_conv_same, NT2_SIMD_REAL_TYPES )
{
  std::size_t data_min= args("min",100);
  std::size_t data_max= args("max",4096);
  std::size_t data_step= args("step",2);
  std::size_t filt_size= args("filter",size_filter);

  run_during_with< test_conv_same<T> > ( 1.
                                    , and_( geometric(data_min,data_max,data_step)
                                          , constant(filt_size)
                                          )
                                    , cycles_per_element<stats::median_>()
                                    );
}

NT2_REGISTER_BENCHMARK_TPL( test_conv_same_static, NT2_SIMD_REAL_TYPES )
{
  std::size_t data_min= args("min",100);
  std::size_t data_max= args("max",4096);
  std::size_t data_step= args("step",2);

  run_during_with< test_conv_same_static<T> > ( 1.
                                    , geometric(data_min,data_max,data_step)
                                    , cycles_per_element<stats::median_>()
                                    );
}

//FULL

NT2_REGISTER_BENCHMARK_TPL( test_conv_full, NT2_SIMD_REAL_TYPES )
{
  std::size_t data_min= args("min",100);
  std::size_t data_max= args("max",4096);
  std::size_t data_step= args("step",2);
  std::size_t filt_size= args("filter",size_filter);

  run_during_with< test_conv_full<T> > ( 1.
                                    , and_( geometric(data_min,data_max,data_step)
                                          , constant(filt_size)
                                          )
                                    , cycles_per_element<stats::median_>()
                                    );
}

NT2_REGISTER_BENCHMARK_TPL( test_conv_full_static, NT2_SIMD_REAL_TYPES )
{
  std::size_t data_min= args("min",100);
  std::size_t data_max= args("max",4096);
  std::size_t data_step= args("step",2);

  run_during_with< test_conv_full_static<T> > ( 1.
                                    , geometric(data_min,data_max,data_step)
                                    , cycles_per_element<stats::median_>()
                                    );
}
