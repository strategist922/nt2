//==============================================================================
//         Copyright 2009 - 2013 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2014 MetaScale SAS
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================

#include <boost/simd/sdk/simd/pack.hpp>
#include <boost/simd/sdk/simd/native.hpp>
#include <nt2/include/functions/plus.hpp>
#include <nt2/include/functions/log.hpp>
#include <nt2/include/functions/exp.hpp>
#include <nt2/include/functions/fastnormcdf.hpp>
#include <boost/simd/include/functions/fma.hpp>
#include <boost/simd/include/functions/fnms.hpp>
#include <boost/simd/include/functions/sqrt.hpp>
#include <boost/simd/include/functions/sqr.hpp>
#include <boost/simd/include/functions/sum.hpp>
#include <boost/simd/include/functions/aligned_store.hpp>
#include <boost/simd/include/functions/aligned_load.hpp>
#include <boost/simd/include/functions/divides.hpp>
#include <boost/simd/include/functions/multiplies.hpp>
#include <boost/simd/include/functions/unary_minus.hpp>
#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/minus.hpp>
#include <boost/simd/include/constants/half.hpp>
#include <boost/simd/memory/allocator.hpp>
#include <vector>
#include <cstdio>

#include <nt2/sdk/bench/benchmark.hpp>
#include <nt2/sdk/bench/metric/cycles_per_element.hpp>
#include <nt2/sdk/bench/protocol/max_duration.hpp>
#include <nt2/sdk/bench/setup/geometric.hpp>
#include <nt2/sdk/bench/stats/median.hpp>

using namespace nt2::bench;
using namespace nt2;

template<typename T> struct sum_simd
{
  sum_simd(std::size_t n)
  :size_(n)
  {
    A.resize(size_*size_);
    B.resize(size_*size_);
    C.resize(size_*size_);
    D.resize(size_*size_);
    tmp.resize(size_);

    for(std::size_t i = 0; i <size_*size_; ++i)
      A[i] = B[i] = C[i] = D[i] = T(i);
  }

  void operator()()
  {
    using boost::simd::pack;
    using boost::simd::native;
    using boost::simd::aligned_store;
    using boost::simd::aligned_load;
    using boost::simd::sum;

    typedef pack<T> type;
    std::size_t step_size_= boost::simd::meta::cardinal_of<type>::value;

    for (std::size_t jj=0, offset=0; jj<size_; jj++, offset+=size_)
    {
      type reduced_value(0);

      for (std::size_t ii=0, k=offset; ii<size_; ii+=step_size_, k+=step_size_)
      {
        type C_tmp = aligned_load<type>(&C[k]);
        type D_tmp = aligned_load<type>(&D[k]);

        reduced_value = reduced_value + C_tmp + D_tmp;
      }

      tmp[jj] = sum(reduced_value);
    }

    for (std::size_t jj=0, offset=0; jj<size_; jj++, offset+=size_)
    {
      for (std::size_t ii=0, k=offset; ii<size_; ii+=step_size_, k+=step_size_)
      {
        type A_tmp = aligned_load<type>(&A[k]);
        type B_tmp = aligned_load<type>(&B[k]);

        A_tmp = B_tmp/tmp[jj];
      }
    }
  }

  friend std::ostream& operator<<(std::ostream& os, sum_simd<T> const& p)
  {
    return os << "(" << p.size() << ")";
  }

  std::size_t size() const { return size_*size_; }

  private:
  std::vector<T, boost::simd::allocator<T> > A, B, C, D, tmp;
  std::size_t size_;
};

NT2_REGISTER_BENCHMARK_TPL( sum_simd, (float) )
{
  std::size_t size_min  = args("size_min" , 8000);
  std::size_t size_max  = args("size_max" , 32000);
  std::size_t size_step = args("size_step",    2);

  run_during_with< sum_simd<float> > ( 10.
                                     , geometric(size_min,size_max,size_step)
                                     , cycles_per_element<stats::median_>()
                                     );
}
