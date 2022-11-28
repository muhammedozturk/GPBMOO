![overview](https://github.com/muhammedozturk/GPBMOO/blob/main/overview.png)
# Gauss Pareto-based Multi-objective Optimization (GPBMOO)
is a multi-objective optimization method coded with R package. It finds the best configuration of Spark in compliance with three fitness functions. GPBMOO has been tested on the benchmarks of Hibench and MLlib library of Spark. sparktf and sparklyr are needed to connect Spark.

##The main advantage of GPBMOO over the alternatives are as follows:
1. Different from preceding works, GPBMOO, which consists of three objective functions, is the first Kriging-based method developed for Spark,
2. GPBMOO can be adapted for Hibench benchmarks, but it is also suitable for other algorithms of MLlib,
3. To achieve high speedup, increasing the number of CPU cores is redundant after a specific threshold in multiobjective optimization

## Connection
![con1](https://github.com/muhammedozturk/GPBMOO/blob/main/con1.png)


## Three fitness functions <br /> 
MOPSPecial <- function (x)  <br /> 
{ <br /> 

  Memory <-1+ (1+cos(x)+x*x-2*x)+ sparkMem(x) <br /> 
  Time <- 1+ (1+sin(x)+x*x-2*sin(x)*x)+sparkTime(x) <br /> 
 Accuracy <- 1+ (1+sin(x)+x*x-2*cos(x)*x)+sparkAccu(x) <br /> 
  f <- cbind(Memory,Time,Accuracy) <br /> 

} <br /> 

## How to run GPBMOO <br /> 

1. First download a data set from https://www.kaggle.com/muhammad4hmed/malwaremicrosoftbig <br /> 
2. Run mopSpecialMicrosoft (1) in R <br /> 

##The following codes creates training and testing parts. <br /> 
######################### <br /> 
ydata <- read.csv("LargeTrain.csv") <br /> 
mydata$Class2 <- ifelse(mydata$Class2 == 1, 'no', 'yes')  <br /> 
mydata <- mydata[,c(1,2,3,4,1805)] <br /> 
rangeRandom <- sample(1:10865,2000) <br /> 
train <- mydata[-rangeRandom,] <br /> 
test <- mydata[rangeRandom,] <br /> 

train <- copy_to(sc, train) <br /> 
test <- copy_to(sc,test) <br /> 
######################################### <br /> 

##The following function is used to configure feed forward neural network. <br /> 
####################################### <br /> 
sparkMem <- function(y){ <br /> 
  nn_model <- ml_multilayer_perceptron_classifier( <br /> 
    train, <br /> 
    Class2 ~ Virtual+Offset+loc+Import , <br /> 
    layers = c(4,3,3,3,2),step_size=y,max_iter=1,tol=0.0066, <br /> 
    solver = "gd") <br /> 
  
  sonuc <- nn_model %>% <br /> 
    ml_predict(test) %>% <br /> 
    select(Class2, predicted_label,starts_with("probability_")) %>% <br /> 
    glimpse() <br /> 
  
  
  frame <- data.frame(sonuc[1],sonuc[2]) <br /> 
  #frame[1,2]="yes"  <br /> 
  #tablo <- table(frame$Class2,frame$predicted_label) <br /> 
  u <- union(frame$predicted_label, frame$Class2) <br /> 
  tablo <- table(factor(frame$predicted_label, u), factor(frame$Class2, u)) <br /> 
  result <- confusionMatrix(tablo) <br /> 
  #################################
  memTotal <- mem_used() <br /> 
  return (memTotal*(y/y)) <br /> 
  #result <- result[3]$overall[1] <br /> 
  #return (-result*(x/x)) <br /> 
}#end of function <br /> 
######################### <br /> 
##The following function is used to conduct three-objectives optimization <br /> 
MOPSPecial <- function (x)  <br /> 
{ <br /> 

  Memory <-1+ (1+cos(x)+x*x-2*x)+ sparkMem(x) <br /> 
  Time <- 1+ (1+sin(x)+x*x-2*sin(x)*x)+sparkTime(x) <br /> 
 Accuracy <- 1+ (1+sin(x)+x*x-2*cos(x)*x)+sparkAccu(x) <br /> 
  #f1 <- 1+ (1+cos(x)+x*x-2*x) <br /> 
  #f2 <- 1+ (1+sin(x)+x*x-2*sin(x)*x) <br /> 
 # f3 <- 1+ (1+sin(x)+x*x-2*cos(x)*x) <br /> 
  f <- cbind(Memory,Time,Accuracy) <br /> 

} <br /> 
##The following line executes the optimization with boundary values <br /> 
res <- easyGParetoptim(fn = MOPSPecial, budget = 50, lower = 0.05,upper = 2,control = list(method = "EHI", trace = 1, inneroptim = "pso", maxit = 5, seed = 42)) <br /> 
