# Gauss Pareto-based Multi-objective Optimization (GPBMOO)
is a multi-objective optimization method coded with R package. It finds the best configuration of Spark in compliance with three fitness functions. GPBMOO has been tested on the benchmarks of Hibench and MLlib library of Spark. sparktf and sparklyr are needed to connect Spark.

##The main advantage of GPBMOO over the alternatives are as follows:
1. Different from preceding works, GPBMOO, which consists of three objective functions, is the first Kriging-based method developed for Spark,
2. GPBMOO can be adapted for Hibench benchmarks, but it is also suitable for other algorithms of MLlib,
3. To achieve high speedup, increasing the number of CPU cores is redundant after a specific threshold in multiobjective optimization

## Connections
library(sparklyr)
library(dplyr)
library(modeldata)
library("caret")
library("base")
library("DEoptim")
library("pryr")
library("peakRAM")
######################################

config <- spark_config()
config["spark.sql.shuffle.partitions"] <- 10
config["sparklyr.connect.cores.local"] <- 100

sc <- spark_connect(master = "local",config=config)
sdf_len(sc, 10, repartition = 10) %>% sdf_num_partitions()

##Fitness Functions
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
#######################################
sparkTime <- function(y){

  ##########RS########################
  #################################
  timeResult <-  system.time({
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
  })
  return (timeResult[3]*(y/y))
  #result <- result[3]$overall[1]
  #return (-result*(x/x))
}#end of function
###########################################
sparkAccu <- function(y){

  ##########RS########################
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
  
  
  result <- result[3]$overall[1]
  return (-result)
}#end of funct

##
