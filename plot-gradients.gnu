set terminal postscript color
unset key
set autoscale y

avg=0.3
alpha=0.01
mkavg(x)=(avg=alpha*x+(1-alpha)*avg,avg)
set output 'fraction.eps'
plot 'log/recent/robust.log' using 1:($4>=$2?mkavg($7):1/0) pt 7 ps 1

#set multiplot layout 2,1 rowsfirst

#unset logscale y
set logscale y
#plot 'log/recent/batch.txt' using 1:($2==0?$4:1/0) lc rgb '#00ff00' pt 1 ps 1,\
     #'log/recent/batch.txt' using 1:($2==0?1/0:$4) lc rgb '#ff0000' pt 7 ps 1,\
     #'log/recent/batch.txt' using 1:5 with lines,\
     #'log/recent/batch.txt' using 1:6 with lines


set output 'gradients.eps'
plot 'log/recent/robust.log' using 1:($7==0?1/0:$4) lc rgb '#ff0000' pt 7 ps 1,\
     'log/recent/robust.log' using 1:($7==0?$4:1/0) lc rgb '#00ff00' pt 1 ps 1,\
     'log/recent/robust.log' using 1:3 with lines,\
     'log/recent/robust.log' using 1:2 with lines
     #'log/recent/robust.log' using 1:($7==0?$4:1/0) lc rgb '#00ff00' pt 1 ps 1,\
#plot 'log/recent/robust.log' using 1:($7==0?$4:1/0) lc rgb '#00ff00' pt 1 ps 1,\
     #'log/recent/robust.log' using 1:($7==1?1/0:$4) lc rgb '#ff0000' pt 7 ps 1,\
#plot 'log/recent/robust.log' using 1:2 with lines,\
     #'log/recent/robust.log' using 1:3 with lines

#set yrange [0:1.1]
#set logscale y
#plot 'log/recent/batch.txt' every 1000::999 using 1:6 with lines,\
     #'log/recent/epoch.txt' using ($1*1000):3 with lines

#unset multiplot
