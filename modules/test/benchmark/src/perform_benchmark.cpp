//==============================================================================
//         Copyright 2003 - 2012 LASMEA UMR 6602 CNRS/Univ. Clermont II
//         Copyright 2009 - 2012 LRI    UMR 8623 CNRS/Univ Paris Sud XI
//         Copyright 2012 - 2013 MetaScale SAS
//         Copyright 2012        Domagoj Saric, Little Endian Ltd.
//
//          Distributed under the Boost Software License, Version 1.0.
//                 See accompanying file LICENSE.txt or copy at
//                     http://www.boost.org/LICENSE_1_0.txt
//==============================================================================
#include <nt2/sdk/bench/config.hpp>
#include <nt2/sdk/bench/details/median.hpp>
#include <nt2/sdk/bench/details/overheads.hpp>
#include <nt2/sdk/bench/perform_benchmark.hpp>
#include <nt2/sdk/timing/now.hpp>
#include <vector>

#include <boost/simd/sdk/config/os.hpp>
#ifdef BOOST_SIMD_OS_LINUX
#include <boost/dispatch/meta/ignore_unused.hpp>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <dirent.h>
#include <errno.h>
#include <map>
#include <string>
#include <stdio.h>
#endif

namespace nt2
{
  namespace details
  {
    #ifdef BOOST_SIMD_OS_LINUX
    struct ensure_max_cpu_freq
    {
      ensure_max_cpu_freq()
      {
        DIR* dir = ::opendir("/sys/devices/system/cpu");
        struct dirent* it;
        while((it = ::readdir(dir)))
        {
          if(it->d_type != DT_DIR || strncmp(it->d_name, "cpu", 3))
            continue;

          std::string path = "/sys/devices/system/cpu/";
          path += it->d_name;
          path += "/cpufreq/scaling_governor";

          int fd = ::open(path.c_str(), O_RDWR);
          if(fd < 0)
          {
            if(errno != ENOENT && errno != EACCES)
              perror(("failure to open scaling governor " + path).c_str());
            continue;
          }

          char buffer[256];
          ssize_t sz = ::read(fd, buffer, sizeof buffer);
          if(sz < 0)
          {
            perror("failure to read scaling governor");
            ::close(fd);
            continue;
          }
          previous_settings[path] = std::string(buffer, buffer+sz);


          ::lseek(fd, 0, SEEK_SET);
          sz = ::write(fd, "performance", 11);
          boost::dispatch::ignore_unused(sz);
          ::close(fd);
        }
      }

      ~ensure_max_cpu_freq()
      {
        for(std::map<std::string, std::string>::iterator it = previous_settings.begin(); it != previous_settings.end(); ++it)
        {
          int fd = ::open(it->first.c_str(), O_WRONLY);
          if(fd < 0)
            continue;

          ssize_t sz = ::write(fd, it->second.c_str(), it->second.size());
          boost::dispatch::ignore_unused(sz);
          ::close(fd);
        }
      }

      std::map<std::string, std::string> previous_settings;
    };
    #else
    struct ensure_max_cpu_freq
    {
      ensure_max_cpu_freq()
      {
      }
    };
    #endif

    static std::vector<nt2::cycles_t      > individual_measurement_cycles;
    static std::vector<nt2::time_quantum_t> individual_measurement_time_quantums;

    NT2_TEST_BENCHMARK_DECL benchmark_result_t reference_timing_value;

    BOOST_DISPATCH_NOINLINE intermediate_result_t
    perform_benchmark_impl( base_experiment const& test, nt2::seconds_t const d )
    {
      nt2::details::ensure_max_cpu_freq ensure_max_cpu_freq_scoped;

      individual_measurement_cycles       .clear();
      individual_measurement_time_quantums.clear();

      time_quantum_t const total_duration( to_timequantums( d * 1000000 ) );
      time_quantum_t       duration      (0);

      test.reset();
      while ( duration < total_duration )
      {
        /// \todo Consider flushing the CPU cache(s) here (the pipeline
        /// presumably gets flushed this loop's code).
        ///                                   (05.11.2012.) (Domagoj Saric)

        time_quantum_t const time_start  ( time_quantum() );
        cycles_t       const cycles_start( read_cycles() );
        test.run();
        cycles_t       const cycles_end( read_cycles() );
        time_quantum_t const time_end  ( time_quantum() );

        cycles_t       const burned_cycles( cycles_end - cycles_start );
        time_quantum_t const elapsed_time ( time_end   - time_start   );

        duration += elapsed_time;

        individual_measurement_cycles       .push_back( burned_cycles );
        individual_measurement_time_quantums.push_back( elapsed_time  );

        test.reset();
      }

      return intermediate_result_t( median(individual_measurement_cycles)
                                  , median(individual_measurement_time_quantums)
                                  );
    }
  }

  NT2_TEST_BENCHMARK_DECL BOOST_DISPATCH_NOINLINE
  benchmark_result_t perform_benchmark( details::base_experiment const& test
                                      , nt2::seconds_t const duration
                                      )
  {
    intermediate_result_t const
    irs( details::perform_benchmark_impl( test, duration ) );

    return  benchmark_result_t
            ( irs.first - details::cycles_overhead
            , to_microseconds(irs.second - details::quantums_overhead)
            );
  }

  NT2_TEST_BENCHMARK_DECL BOOST_DISPATCH_NOINLINE
  benchmark_result_t& reference_timing()
  {
    return details::reference_timing_value;
  }
}
