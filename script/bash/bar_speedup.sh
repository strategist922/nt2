#!/bin/bash

if [ $# -lt 1 ]
then
    echo "Usage: `basename $0` <program>..."
    echo "If program is -, use content of existing res.txt file"
    echo "RUN_CMD may be defined to an arbitrary prefix command"
    echo "RUN_ARGS may be defined to an arbitrary list of arguments"
    exit 65
fi

# run all programs, filter undesired output
if [ "$1" != "-" ]
then
  rm -f res.txt
  for i in "$@"
  do
    $RUN_CMD $i $RUN_ARGS | stdbuf -o0 grep -Ev "CTEST_FULL_OUTPUT|---" | stdbuf -o0 tail -n +2 | tee -a res.txt
  done
fi

# count number of experiments and sizes
# validate number of sizes
# compute longest common prefix
prefix=
prev=
prev_nb_sizes=0
nb_sizes=0
NUM_TEST=0
while read l
do
  p=$(echo "$l" | cut -d$'\t' -f1)
  if [ "$prev" == "" ] || [ "$prev" != "$p" ]
  then
    echo $p
    if [ $nb_sizes -ne $prev_nb_sizes ] && [ $prev_nb_sizes -ne 0 ]
    then
      echo "inconsistent number of sizes, has $nb_sizes, previous had $prev_nb_sizes" >&2
      exit 1
    fi
    prev_nb_sizes=$nb_sizes
    nb_sizes=0
    NUM_TEST=$(($NUM_TEST+1))
  fi
  prev=$p
  nb_sizes=$(($nb_sizes+1))

  if [ "$prefix" == "" ]
  then
    prefix=$p
  else
    prefix=$(printf "%s\n%s\n" $prefix $p | sed -e 'N;s/^\(.*\).*\n\1.*$/\1/')
  fi
done < res.txt > exp.txt
if [ $nb_sizes -ne $prev_nb_sizes ] && [ $prev_nb_sizes -ne 0 ]
then
   echo "inconsistent number of sizes, has $nb_sizes, previous had $prev_nb_sizes" >&2
   exit 1
fi

# benchmark name is the longest prefix of all experiments without the last _
BENCH_NAME=$(echo "$prefix" | sed -r 's/_[^_]+//')
TITLE="title(\"$BENCH_NAME}\")"

# compute legend
LEGEND=""
while read p
do
  p2=$(echo "$p" | sed -r "s/^${BENCH_NAME}_//")
  if [ "$LEGEND" == "" ]
  then
    LEGEND="legend(\"$p2\""
  else
    LEGEND="$LEGEND, \"$p2\""
  fi
done < exp.txt
LEGEND="$LEGEND)"

# generate M-file
i=0
while read p; do
  SIZE[i]=$(echo $p | tr "(" "\n" | tr ")" "\n" | sed -n 2p)
  CYC[i]=$(echo $p | awk  '{ print $3 }')
  UNIT=$(echo $p | awk '{print $4 }')
  ((i++))
done < res.txt

for ((ii=10#0;ii<${nb_sizes};ii++));do
  for ((jj=10#0;jj<(${NUM_TEST});jj++));do
    q=(${ii}+${jj}*${nb_sizes})
    CYC2[${jj}+${ii}*(${NUM_TEST})]=${CYC[q]}
  done
done

gap=5
((size_x_axis=${gap}*(${nb_sizes}+1)+(${NUM_TEST})*${nb_sizes}))
cyc='cyc=['

echo "hold on" > create_plot.m

X="x=["

for ((i = 1 ; i <= ${size_x_axis} ; i++)); do
  X="$X $i"
done

for ((ii=0;ii<${nb_sizes};ii++));do
  for ((jj=10#0;jj<${gap}+(${NUM_TEST});jj++));do
    if ((jj<${gap}));then
      cyc="${cyc} 0"
    else
      ((q= (${jj}-${gap}) + ${ii}*(${NUM_TEST})))
      cyc="${cyc} ${CYC2[${q}]}"
    fi
  done
done

for ((ii=0;ii<${gap};ii++));do
  cyc="${cyc} 0"
done
cyc="${cyc} ];"

echo ${cyc} >> create_plot.m
echo ${TITLE}  >> create_plot.m

X="$X ];"
echo $X>>create_plot.m

echo "num_y_ticks = round(max(cyc))">>create_plot.m
((q=${gap}+1))
echo "num_div=1;">>create_plot.m

echo "max_y=max(cyc)*1.2">>create_plot.m
echo "y_inc = max_y/num_y_ticks">>create_plot.m
echo "x_points=ones(1,num_y_ticks);">>create_plot.m
echo "y_points=zeros(1,num_y_ticks);">>create_plot.m
echo "  y_dist=5;">>create_plot.m
echo "if (max(cyc)>1000)">>create_plot.m
echo "  num_div=100;">>create_plot.m
echo "  y_dist=5;">>create_plot.m
echo "elseif (max(cyc)>500)">>create_plot.m
echo "  num_div=50;">>create_plot.m
echo "  y_dist=4;">>create_plot.m
echo "elseif (max(cyc)>160)">>create_plot.m
echo "  num_div=16;">>create_plot.m
echo "  y_dist=4;">>create_plot.m
echo "elseif (max(cyc)>80)">>create_plot.m
echo "  num_div=8;">>create_plot.m
echo "elseif (max(cyc)>40)">>create_plot.m
echo "  num_div=4;">>create_plot.m
echo "elseif (max(cyc)>20)">>create_plot.m
echo "  num_div=2;">>create_plot.m
echo "elseif (max(cyc)>10)">>create_plot.m
echo "  num_div=1;">>create_plot.m
echo "endif">>create_plot.m
echo "num_y_ticks/=num_div;">>create_plot.m
echo "y_points(1)=y_inc*num_div;">>create_plot.m
echo "for i=2:num_y_ticks">>create_plot.m
echo "  y_points(i)=y_points(i-1)+y_inc*num_div;">>create_plot.m
echo "endfor">>create_plot.m

echo "w=0.4;">>create_plot.m

COLORS[0]=" 'r' "
COLORS[1]=" 'g' "
COLORS[2]=" 'b' "
COLORS[3]=" 'c' "
COLORS[4]=" 'y' "
COLORS[5]=" \"4\" "
COLORS[6]=" \"5\" "
COLORS[7]=" \"6\" "
COLORS[8]=" 'r' "
COLORS[9]=" 'g' "
COLORS[10]=" 'b' "
j=0
for ((ii=gap;ii<${size_x_axis};ii+=${gap}+(${NUM_TEST})));do
  for ((jj=10#;jj<(${NUM_TEST});jj++));do
    ((q=${ii}+1+${jj}))
    echo "bar(x(${q}),cyc(${q}),w,${COLORS[jj]});">>create_plot.m
  done
done

for ((ii=0;ii<${nb_sizes};ii++));do
  for ((jj=${gap}+1;jj<${gap}+(${NUM_TEST});jj++));do
    ((q= ${jj} + 1 + ${ii}*(${gap}+(${NUM_TEST}))))
    ((r=(${ii})*(${gap}+(${NUM_TEST}))+${gap}+1))
    if [ "$UNIT" ==  "GFLOPS" ];then
      echo "t=strcat(\"x\",mat2str(cyc(${q})/cyc(${r}),2));">>create_plot.m
    else
      echo "t=strcat(\"x\",mat2str(cyc(${r})/cyc(${q}),2));">>create_plot.m
    fi
    echo "H=text(x(${q}),cyc(${q})+num_div*0.05,t,\"rotation\",90)">>create_plot.m
    echo "set(H,'fontsize',7)">>create_plot.m
  done
done

echo "max_x = max(x)+${gap};">>create_plot.m

echo ${LEGEND}>>create_plot.m
echo "xx=[1 max_x max_x 1 1];">>create_plot.m
echo "text(-4,max_y/2-num_div,\"${UNIT}\",\"rotation\",90)">>create_plot.m
echo "yy=[0 0 max_y max_y 0];">>create_plot.m
echo "plot(xx,yy,'k')">>create_plot.m
echo "ylim([-num_div*0.5 max_y])">>create_plot.m
echo "axis off" >>create_plot.m
echo "xlim([-4 max_x]);">>create_plot.m

for ((ii=1;ii<=${nb_sizes};ii++));do
  ((q=${gap}+(${ii}-1)*(${gap}+(${NUM_TEST}))))
  echo "text(x(${q}),-num_div*0.3,\"${SIZE[ii-1]}\")">>create_plot.m
done

echo "box on" >>create_plot.m
echo "plot(x_points,y_points,'+k')" >>create_plot.m
echo "for i=1:num_y_ticks" >>create_plot.m
echo "  text(x_points(i)-y_dist,y_points(i),mat2str(i*num_div));" >>create_plot.m
echo "endfor" >>create_plot.m
echo "print ('-dpng','-color',\"${BENCH_NAME}.png\")">>create_plot.m
echo "h = figure;">>create_plot.m
echo "W = 12; H = 3;">>create_plot.m
echo "set(h,'PaperUnits','inches')">>create_plot.m
echo "set(h,'PaperOrientation','portrait');">>create_plot.m
echo "set(h,'PaperSize',[H,W])">>create_plot.m
echo "set(h,'PaperPosition',[0,0,W,H])">>create_plot.m
echo "refresh">>create_plot.m

octave create_plot.m &>  /dev/null; display "${BENCH_NAME}.png" &
