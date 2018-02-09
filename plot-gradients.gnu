corrupt=10

#set logscale y

#set multiplot layout 2,1 rowsfirst

#unset logscale y
set logscale y
set autoscale y
plot 'log/recent/batch.txt' using 1:($2==0?$4:1/0) lc rgb '#00ff00' pt 1 ps 1,\
     'log/recent/batch.txt' using 1:($2==0?1/0:$4) lc rgb '#ff0000' pt 7 ps 1,\
     'log/recent/batch.txt' using 1:5 with lines,\
     'log/recent/batch.txt' using 1:6 with lines

#plot 'log/recent/batch.txt' using 1:($3>=corrupt?$4:1/0) lc rgb '#00ff00' pt 1 ps 1,\
     #'log/recent/batch.txt' using 1:($3>=corrupt?1/0:$4) lc rgb '#ff0000' pt 7 ps 1,\
     #'log/recent/batch.txt' using 1:5 with lines,\
     #'log/recent/batch.txt' using 1:6 with lines

#set yrange [0:1.1]
#set logscale y
#plot 'log/recent/batch.txt' every 1000::999 using 1:6 with lines,\
     #'log/recent/epoch.txt' using ($1*1000):3 with lines

#unset multiplot
