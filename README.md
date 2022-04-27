# Gauss Pareto-based Multi-objective Optimization (GPBMOO)
is a multi-objective optimization method coded with R package. It finds the best configuration of Spark in compliance with three fitness functions. GPBMOO has been tested on the benchmarks of Hibench and MLlib library of Spark. sparktf and sparklyr are needed to connect Spark.

##The main advantage of GPBMOO over the alternatives are as follows:
1. Different from preceding works, GPBMOO, which consists of three objective functions, is the first Kriging-based method developed for Spark,
2. GPBMOO can be adapted for Hibench benchmarks, but it is also suitable for other algorithms of MLlib,
3. To achieve high speedup, increasing the number of CPU cores is redundant after a specific threshold in multiobjective optimization

## Three fitness functions
MOPSPecial <- function (x) 
{

  Memory <-1+ (1+cos(x)+x*x-2*x)+ sparkMem(x)
  Time <- 1+ (1+sin(x)+x*x-2*sin(x)*x)+sparkTime(x)
 Accuracy <- 1+ (1+sin(x)+x*x-2*cos(x)*x)+sparkAccu(x)
  f <- cbind(Memory,Time,Accuracy)

}
![con1](https://github.com/muhammedozturk/GPBMOO/blob/main/con1.png)
