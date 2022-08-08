#=================================
# install packages
#=================================
  
install.packages("Amelia")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("reshape")
install.packages("randomForest")
install.packages("e1071")

# =================================
# workspace preparation
#=================================

setwd("<your current path>")
getwd()


#=================================
# packages declaration
#=================================

library(Amelia)
library(caret)
library(ggplot2)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)


#=================================
# General of data view
#=================================

dados_clientes <- read.csv("dataset.csv")
View(dados_clientes)
dim(dados_clientes)
str(dados_clientes)
summary(dados_clientes)


#=================================
# Analysis, cleaning and transformation
#=================================


# remove first column id
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

#rename class column
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "Inadimplente"
colnames(dados_clientes)
View(dados_clientes)

# verify missing values
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main="Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes) #omite dados ausente

#=================================
# Convert Categorical / Qualitative Values
#=================================


# rename categorical columns
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "Genero"
colnames(dados_clientes)[3] <- "Escolaridade"
colnames(dados_clientes)[4] <- "Estado_Civil"
colnames(dados_clientes)[5] <- "Idade"
colnames(dados_clientes)
View(dados_clientes)


# Genre
View(dados_clientes$Genero)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)
?cut
dados_clientes$Genero <- cut(dados_clientes$Genero, 
                             c(0,1,2),
                             labels = c("Masculino",
                                        "Feminino"))
View(dados_clientes)
str(dados_clientes$Genero)
summary(dados_clientes$Genero)

# Education
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)
dados_clientes$Escolaridade <- cut(dados_clientes$Escolaridade, 
                                   c(0,1,2,3,4),
                                   labels = c("Pos Graduado",
                                              "Graduado",
                                              "Ensino Medio",
                                              "Outros"))
View(dados_clientes$Escolaridade)
str(dados_clientes$Escolaridade)
summary(dados_clientes$Escolaridade)


# Marital Status

str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)
dados_clientes$Estado_Civil <- cut(dados_clientes$Estado_Civil, 
                                   c(-1, 0, 1, 2, 3),
                                   labels = c("Desconhecido",
                                              "Casado",
                                              "Solteiro",
                                              "Outro"))
View(dados_clientes$Estado_Civil)
str(dados_clientes$Estado_Civil)
summary(dados_clientes$Estado_Civil)


# convert age to age group
str(dados_clientes$Idade)
summary(dados_clientes$Idade)
dados_clientes$Idade <- cut(dados_clientes$Idade, 
                                   c(0, 30, 50, 100),
                                   labels = c("Jovem",
                                              "Adulto",
                                              "Idoso"))
View(dados_clientes$Idade)
str(dados_clientes$Idade)
summary(dados_clientes$Idade)



# converting the variables that indicate payment to factor
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# dataset after conversions
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main="Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes) #omite dados ausente
dim(dados_clientes)

# Changing the default variable to factor

dados_clientes$Inadimplente <- as.factor(dados_clientes$Inadimplente)
str(dados_clientes)


#=================================
# Prepare training and test samples
#=================================



# check the total number of defaulters
table(dados_clientes$Inadimplente)
prop.table(table(dados_clientes$Inadimplente))


#set seed
set.seed(12345)


# stratified sample
# Selects the lines according to the default variable as stratified
?createDataPartition
indice <- createDataPartition(dados_clientes$Inadimplente, p=0.75, list=FALSE)
dim(indice)
View(indice)


# set training data
dados_treino <- dados_clientes[indice,]
dim(dados_treino)
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))

# set the test data
dados_teste <- dados_clientes[-indice,]
dim(dados_teste)
table(dados_teste$Inadimplente)
prop.table(table(dados_teste$Inadimplente))




#=================================
# Machine Learning Model v1 - Unbalanced Model
#=================================

# Build first version of the model
?randomForest
modelo_v1 <- randomForest(Inadimplente ~ ., data = dados_treino)
modelo_v1

# evaluate the model
plot(modelo_v1)

# Forecasts with the model
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix to evaluate predictions
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$Inadimplente, positive="1")
cm_v1

# calculating Precision, Recall and F1-Score - metrics for evaluating predictive models
y <- dados_teste$Inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1,y)
recall <- sensitivity(y_pred_v1,y)
F1 <- (2 * precision * recall) / (precision + recall)

precision
recall
F1


#=================================
# Machine Learning Model v2 - Balanced Model
#=================================

# class balancing
install.packages(c("zoo","xts","quantmod", "abind", "ROCR"))
install.packages( "https://cran.r-project.org/src/contrib/Archive/DMwR/DMwR_0.4.1.tar.gz", repos=NULL, type="source" )
library(DMwR)
?SMOTE

# aplicando SMOTE
table(dados_treino$Inadimplente)
prop.table(table(dados_treino$Inadimplente))
set.seed(9560)
dados_treino_bal <- SMOTE(Inadimplente ~ ., data = dados_treino)
table(dados_treino_bal$Inadimplente)
prop.table(table(dados_treino_bal$Inadimplente))


# Build second version of the model
modelo_v2 <- randomForest(Inadimplente ~ ., data = dados_treino_bal)
modelo_v2


# evaluate the model
plot(modelo_v2)

# Forecasts with the model
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Confusion Matrix to evaluate predictions
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$Inadimplente, positive="1")
cm_v2

# calculating Precision, Recall and F1-Score - metrics for evaluating predictive models
y <- dados_teste$Inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2,y)
recall <- sensitivity(y_pred_v2,y)
F1 <- (2 * precision * recall) / (precision + recall)

precision
recall
F1

#=================================
# Machine Learning Model v3 - Balanced Model + Relevant Variables
#=================================

# Importance of predictor variables for forecasts
varImpPlot(modelo_v2)


# getting more important variables
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[ , 'MeanDecreaseGini'],2))

# creating a variable rank based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# ggplot para visualizar a importancia das variaveis
ggplot(rankImportance,
       aes(x = reorder(Variables, Importance),
           y = Importance,
           fill = Importance)) + 
  geom_bar(stat = 'identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            color = 'red') + 
  labs(x = 'Variables') + 
  coord_flip()


# creating V3 model with the main variables
colnames(dados_treino_bal)
modelo_v3 <- randomForest(Inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1, data = dados_treino_bal)
modelo_v3


# evaluate the model
plot(modelo_v3)

# Forecasts with the model
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Confusion Matrix to evaluate predictions
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$Inadimplente, positive="1")
cm_v3

# calculating Precision, Recall and F1-Score - metrics for evaluating predictive models
y <- dados_teste$Inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3,y)
recall <- sensitivity(y_pred_v3,y)
F1 <- (2 * precision * recall) / (precision + recall)

precision
recall
F1


#=================================
# Save template to disk
#=================================


saveRDS(modelo_v3, file="modelo_v3.rds")
modelo_final <-readRDS("modelo_v3.rds")



#=================================
# Previsões com novos dados de 3 clientes
#=================================



# customer data

PAY_0 <- c(0,0,0)
PAY_2 <- c(0,0,0)
PAY_3 <- c(1,0,0)
PAY_AMT1 <- c(1100,1000,1200)
PAY_AMT2 <- c(1500,1300,1150)
PAY_5 <- c(0,0,0)
BILL_AMT1 <- c(350,420,280)


# concatenate into a dataframe
novos_clientes <- data.frame(PAY_0,PAY_2,PAY_3,PAY_AMT1,PAY_AMT2,PAY_5,BILL_AMT1)
View(novos_clientes)

# forecasts
previsoes_novos_clientes <- predict(modelo_final, novos_clientes)
str(novos_clientes)


# convert new customer data
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY5))

str(novos_clientes)

previsoes_novos_clientes <- predict(modelo_final, novos_clientes)



