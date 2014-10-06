#!/bin/bash

tmp="${1}/tmp"
out="${1}/include/nt2/sdk/runtime_costs.hpp"

# Normal .hpp generation
if [ -f $out ]; then
  echo -e "#ifndef NT2_SDK_RUNTIME_COSTS_HPP_INCLUDED" > $tmp
  echo -e "#define NT2_SDK_RUNTIME_COSTS_HPP_INCLUDED\n" >> $tmp
  opt="-t `grep --count processor /proc/cpuinfo`"
  for nt2_arch in "openmp" "hpx" "tbb"
  do
    for skel in "transform" "fold" "scan"
    do
      next_exe="bench/arch.${nt2_arch}.microbench.${skel}.bench"
      echo -n "typedef typename boost::mpl::size_t< (std::size_t)" >> ${tmp}
      if [ -f ${next_exe} ]; then
        echo -n `./${next_exe} ${opt} | grep "cpe" | cut -f 3` >> ${tmp}
      else
        echo -n "0" >> ${tmp}
      fi
      echo -e " > ${nt2_arch}_${skel};" >> ${tmp}
    done
    echo "" >> ${tmp}
  done
  echo -e "\n#endif" >> $tmp
  mv $tmp $out

# First .hpp generation
else
  out="${1}/include_tmp/nt2/sdk/runtime_costs.hpp"
  echo -e "#ifndef NT2_SDK_RUNTIME_COSTS_HPP_INCLUDED" > $out
  echo -e "#define NT2_SDK_RUNTIME_COSTS_HPP_INCLUDED\n" >> $out
  echo -e "#include <boost/mpl/size_t.hpp>\n" >> $out
  for nt2_arch in "openmp" "hpx" "tbb"
  do
    for skel in "transform" "fold" "scan"
    do
      echo "typedef typename boost::mpl::size_t<0> ${nt2_arch}_${skel};" >> ${out}
    done
    echo "" >> ${out}
  done
  echo -e "\n#endif" >> $out
fi
