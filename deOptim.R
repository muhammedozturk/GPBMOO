library(sparktf)
library(sparklyr)
library(dplyr)
library(modeldata)
library("caret")
library("base")
library("DEoptim")
######################################

config <- spark_config()
config["spark.sql.shuffle.partitions"] <- 10


sc <- spark_connect(master = "local",config=config)
sdf_len(sc, 10, repartition = 10) %>% sdf_num_partitions()


mydata <- read.csv("qsar-biodeg.csv")
mydata <- copy_to(sc, mydata)
###count of loop for RS

deOptimSpark <- function(x){
  ##########RS########################
  count <- sample(1:8, 1)
  layers <- rep(3,count)
  layers <- append(layers,4,0)
  layers <- append(layers,2)
  #################################
  
  nn_model <- ml_multilayer_perceptron_classifier(
    mydata,
    Class2 ~ V1+V2+V17+V18,
    layers = c(4,3,3,3,2),step_size=x,max_iter=1,
    solver = "gd")
  
  sonuc <- nn_model %>%
    ml_predict(mydata) %>%
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
  return (-result*(x/x))
}#end of funct
#####################################
# get some run-time on simple problems
system.time({
oneCore <- DEoptim(deOptimSpark, lower=0.001, upper=0.006,DEoptim.control(itermax= 1))
oneCore$member$bestmemit})

#######################
#######################
#######################
#######################
#ALTERNATIVE II
###########################
data("attrition")

attrition <- copy_to(sc, attrition)



deOptimSpark <- function(x){
  ##########RS########################
  count <- sample(1:8, 1)
  layers <- rep(3,count)
  layers <- append(layers,4,0)
  layers <- append(layers,2)
  #################################
  
  nn_model <- ml_multilayer_perceptron_classifier(
    attrition,
    Attrition ~ Age + DailyRate + DistanceFromHome + MonthlyIncome,
    layers = c(4,3,3,3,3,2,2),step_size = 0.005833233,
    solver = "gd")
  
  sonuc <- nn_model %>%
    ml_predict(attrition) %>%
    select(Attrition, predicted_label,starts_with("probability_")) %>%
    glimpse()
  
  
  frame <- data.frame(sonuc[1],sonuc[2])
  #frame[1,2]="yes"  ###Bu sat1r1 yazma amac1m1z sütun 1 adet Yes olsun diye. Hepsi No ç1k1yor ve 
  ###confusion bu durumda hesaplanmaz.
  #tablo <- table(frame$Class2,frame$predicted_label)
  u <- union(frame$predicted_label, frame$Attrition)
  tablo <- table(factor(frame$predicted_label, u), factor(frame$Attrition, u))
  result <- confusionMatrix(tablo)
  #################################
  
  
  result <- result[3]$overall[1]
  return (-result*(x/x))
}#end of function

#####################################
# get some run-time on simple problems

oneCore <- DEoptim(deOptimSpark, lower=0.001, upper=0.006,DEoptim.control(itermax= 1)
oneCore$member$bestmemit
############################################

