![overview](https://github.com/muhammedozturk/GPBMOO/blob/main/overview.png)
# Gauss Pareto-based Multi-objective Optimization (GPBMOO)
is a multi-objective optimization method coded with R package. It finds the best configuration of Spark in compliance with three fitness functions. GPBMOO has been tested on the benchmarks of Hibench and MLlib library of Spark. sparktf and sparklyr are needed to connect Spark.

##The main advantage of GPBMOO over the alternatives are as follows:
1. Different from preceding works, GPBMOO, which consists of three objective functions, is the first Kriging-based method developed for Spark,
2. GPBMOO can be adapted for Hibench benchmarks, but it is also suitable for other algorithms of MLlib,
3. To achieve high speedup, increasing the number of CPU cores is redundant after a specific threshold in multiobjective optimization

## Connection
![con1](https://github.com/muhammedozturk/GPBMOO/blob/main/con1.png)


## Three fitness functions
MOPSPecial <- function (x) 
{

  Memory <-1+ (1+cos(x)+x*x-2*x)+ sparkMem(x)
  Time <- 1+ (1+sin(x)+x*x-2*sin(x)*x)+sparkTime(x)
 Accuracy <- 1+ (1+sin(x)+x*x-2*cos(x)*x)+sparkAccu(x)
  f <- cbind(Memory,Time,Accuracy)

}

## How to run GPBMOO

1. First download a data set from https://www.kaggle.com/muhammad4hmed/malwaremicrosoftbig
2. Run mopSpecialMicrosoft (1) in R

##The following codes creates training and testing parts.
#########################
ydata <- read.csv("LargeTrain.csv")
mydata$Class2 <- ifelse(mydata$Class2 == 1, 'no', 'yes') 
mydata <- mydata[,c(1,2,3,4,1805)]
rangeRandom <- sample(1:10865,2000)
train <- mydata[-rangeRandom,]
test <- mydata[rangeRandom,]

train <- copy_to(sc, train)
test <- copy_to(sc,test)
#########################################

##The following function is used to configure feed forward neural network.
#######################################
sparkMem <- function(y){

  ##########RS########################
  #################################
  nn_model <- ml_multilayer_perceptron_classifier(
    train,
    Class2 ~ Virtual+Offset+loc+Import ,
    layers = c(4,3,3,3,2),step_size=y,max_iter=1,tol=0.0066,
    solver = "gd")
  
  sonuc <- nn_model %>%
    ml_predict(test) %>%
    select(Class2, predicted_label,starts_with("probability_")) %>%
    glimpse()
  
  
  frame <- data.frame(sonuc[1],sonuc[2])
  #frame[1,2]="yes"  ###Bu sat1r1 yazma amac1m1z sütun 1 adet Yes olsun diye. Hepsi No ç1k1yor ve 
  ###confusion bu durumda hesaplanmaz.
  #tablo <- table(frame$Class2,frame$predicted_label)
  u <- union(frame$predicted_label, frame$Class2)
  tablo <- table(factor(frame$predicted_label, u), factor(frame$Class2, u))
  result <- confusionMatrix(tablo)
  #################################
  memTotal <- mem_used()
  return (memTotal*(y/y))
  #result <- result[3]$overall[1]
  #return (-result*(x/x))
}#end of function
#########################
##The following function is used to conduct three-objectives optimization
MOPSPecial <- function (x) 
{

  Memory <-1+ (1+cos(x)+x*x-2*x)+ sparkMem(x)
  Time <- 1+ (1+sin(x)+x*x-2*sin(x)*x)+sparkTime(x)
 Accuracy <- 1+ (1+sin(x)+x*x-2*cos(x)*x)+sparkAccu(x)
  #f1 <- 1+ (1+cos(x)+x*x-2*x)
  #f2 <- 1+ (1+sin(x)+x*x-2*sin(x)*x)
 # f3 <- 1+ (1+sin(x)+x*x-2*cos(x)*x)
  f <- cbind(Memory,Time,Accuracy)

}
##The following line executes the optimization with boundary values
res <- easyGParetoptim(fn = MOPSPecial, budget = 50, lower = 0.05,upper = 2,control = list(method = "EHI", trace = 1, inneroptim = "pso", maxit = 5, seed = 42))
