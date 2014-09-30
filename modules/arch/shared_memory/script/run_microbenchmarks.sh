#!/bin/bash

out="${PWD}/include/nt2/sdk/runtime_costs.hpp"

echo -e "#ifndef NT2_SDK_RUNTIME_COSTS_HPP_INCLUDED" > $out
echo -e "#define NT2_SDK_RUNTIME_COSTS_HPP_INCLUDED\n" >> $out

echo -e "#include <boost/mpl/size_t.hpp>\n" >> $out

opt="-t `grep --count processor /proc/cpuinfo`"

for nt2_arch in "openmp" "hpx" "tbb"
do
  for skel in "transform" "fold" "scan"
  do
    # ninja "arch.${nt2_arch}.microbench.${skel}.bench"
    next_exe="bench/arch.${nt2_arch}.microbench.${skel}.bench"
    if [ -f ${next_exe} ]; then
      echo -n "typedef typename boost::mpl::size_t< (std::size_t)" >> ${out}
      # echo -n `./${next_exe} ${opt} | grep "s\." | cut -f 3` >> ${out}
      echo -n `./${next_exe} ${opt} | grep "cpe" | cut -f 3` >> ${out}
      echo -e " > ${nt2_arch}_${skel};" >> ${out}
    fi
  done
  echo "" >> ${out}
done

echo -e "\n#endif" >> $out
