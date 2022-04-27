#sc %>% spark_session() %>% invoke("catalog") %>% invoke("dropTempView", "train")

library(sparktf)
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

mydata <- read.csv("LargeTrain.csv")
mydata$Class2 <- ifelse(mydata$Class2 == 1, 'no', 'yes') 
mydata <- mydata[,c(1,2,3,4,1805)]
rangeRandom <- sample(1:10865,2000)
train <- mydata[-rangeRandom,]
test <- mydata[rangeRandom,]

train <- copy_to(sc, train)
test <- copy_to(sc,test)


###count of loop for RS
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
#####################################
# get some run-time on simple problems

#oneCore <- DEoptim(deOptimSpark, lower=0.001, upper=0.006,DEoptim.control(itermax= 1))
#oneCore$member$bestmemit

####set.seed(25468)
library(DiceDesign)
library("GPareto")
###original###
######################
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


#design.init <- matrix(seq(1, 500, length.out = 10), ncol = 2)
#response.init <- MOPSPecial(design.init)

############easyGpareto################
res <- easyGParetoptim(fn = MOPSPecial, budget = 50, lower = 0.05,upper = 2,control = list(method = "EHI", trace = 1, inneroptim = "pso", maxit = 5, seed = 42))


####plot part
###############################
par(mar=c(1,1,1,1))
par(mar=c(4,4,1,1))
plotGPareto(res)
title("Pareto Front")
plot(res$history$X, main="Pareto set", col = "red", pch = 20)
points(res$par, col="blue", pch = 17)
lower=rep(0,1)
upper=rep(1,1)
plot_uncertainty(res$model,lower=lower,upper=upper)
###ikinci alternatif plotr
library("rgl", quietly = TRUE)
library("knitr")
knit_hooks$set(webgl=hook_webgl)
r3dDefaults$windowRect <- c(0,50, 800, 800) # for better looking figure
plotGPareto(res, UQ_PS = TRUE, control = list(lower = rep(0, 4), upper = rep(1, 4), nintegpoints = 100, option = "mean",resolution = 25))
rgl.postscript("largeTrain.eps","eps")