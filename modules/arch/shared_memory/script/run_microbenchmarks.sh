#!/bin/bash

tmp="${1}/tmp"
out="${1}/include/nt2/sdk/runtime_costs.hpp"

# First .hpp generation
if [ -d "${1}/include_tmp" ]; then
  tmp="${1}/include_tmp/nt2/sdk/runtime_costs.hpp"
# Normal .hpp generation
else
  tmp="${1}/tmp"
fi

echo -e "#ifndef NT2_SDK_RUNTIME_COSTS_HPP_INCLUDED" > $tmp
echo -e "#define NT2_SDK_RUNTIME_COSTS_HPP_INCLUDED\n" >> $tmp
opt="-t `grep --count processor /proc/cpuinfo`"
for nt2_arch in "openmp" "hpx" "tbb"
do
  for skel in "transform" "fold" "scan"
  do
    next_exe="bench/arch.${nt2_arch}.microbench.${skel}.bench"
    echo -n "typedef typename boost::mpl::size_t< " >> ${tmp}
    if [ -f ${next_exe} ]; then
      echo -n "(std::size_t) " >> ${tmp}
      echo -n `./${next_exe} ${opt} | grep "cpe" | cut -f 3` >> ${tmp}
    else
      echo -n "0" >> ${tmp}
    fi
    echo -e " > ${nt2_arch}_${skel};" >> ${tmp}
  done
  echo "" >> ${tmp}
done
echo -e "\n#endif" >> $tmp

if [ ! -d "${1}/include_tmp" ]; then
  mv $tmp $out
fi
