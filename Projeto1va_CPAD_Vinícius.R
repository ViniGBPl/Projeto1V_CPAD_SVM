# --- 1. Instalação e carregamento de pacotes necessários -------------------
if (!require("readr")) install.packages("readr")    # leitura de arquivos
if (!require("dplyr")) install.packages("dplyr")    # manipulação de data.frames
if (!require("e1071")) install.packages("e1071")    # SVM
if (!require("caret")) install.packages("caret")    # treino e avaliação de modelos

library(readr)
library(dplyr)
library(e1071)
library(caret)

# --- 2. Definição de URL e nome do arquivo -------------------------------
url     <- "https://www.dropbox.com/scl/fi/s5it233rz94eg7k3oe9ae/heart.csv?rlkey=nci53drb1kg4irvxeympopyf7&st=82nke1hu&dl=1"
arquivo <- "heart.csv"

# --- 3. Download e leitura dos dados -------------------------------------
download.file(url, destfile = arquivo, mode = "wb")  # baixa o CSV
dados <- read.csv(file = arquivo, 
                  header = TRUE, 
                  strip.white = TRUE, 
                  na.strings = "")    # lê em data.frame

# Inspeção inicial
head(dados)  # primeiras linhas
str(dados)   # estrutura e tipos


dados <- dados[!duplicated(dados), ]  # remove linhas exatamente idênticas

# --- 5. Conversão de variáveis categóricas em fatores --------------------
dados$sex    <- factor(dados$sex,    levels = c(0, 1), labels = c("female", "male"))
dados$fbs    <- factor(dados$fbs,    levels = c(0, 1), labels = c("false", "true"))
dados$thal   <- factor(dados$thal,   levels = c(1, 2, 3), 
                       labels = c("normal", "fixed defect", "reversible defect"))
dados$target <- factor(dados$target, levels = c(0, 1), labels = c("No Disease", "Disease"))
dados$cp     <- factor(dados$cp,     levels = c(0, 1, 2, 3),
                       labels = c("typical angina", "atypical angina", 
                                  "non-anginal pain", "asymptomatic"))
dados$exang  <- factor(dados$exang,  levels = c(0, 1), labels = c("no", "yes"))
dados$restecg<- factor(dados$restecg,levels = c(0, 1, 2),
                       labels = c("Normal", "ST-T abnormality", 
                                  "Left ventricular hypertrophy"))
dados$slope  <- factor(dados$slope,  levels = c(0, 1, 2),
                       labels = c("Downsloping", "Flat", "Upsloping"))

# --- 6. Remoção de valores ausentes --------------------------------------
dados <- na.omit(dados)  # descarta linhas com qualquer NA

# Inspeção pós-tratamento
head(dados)   
str(dados)
summary(dados)  # estatísticas descritivas

# --- 7. Divisão em treino (70%) e teste (30%) ----------------------------
set.seed(123)  # para reproduzir a amostragem
idx    <- sample(seq_len(nrow(dados)), size = 0.7 * nrow(dados))
treino <- dados[idx, ]
teste  <- dados[-idx, ]

# --- 8. Normalização das variáveis numéricas ----------------------------
num_cols     <- sapply(treino, is.numeric)  # identifica colunas numéricas
media_treino <- colMeans(treino[, num_cols])  
sd_treino    <- apply(treino[, num_cols], 2, sd)

treino_n <- treino
teste_n  <- teste
treino_n[, num_cols] <- scale(treino[, num_cols], 
                              center = media_treino, 
                              scale  = sd_treino)
teste_n[,  num_cols] <- scale(teste[,  num_cols], 
                              center = media_treino, 
                              scale  = sd_treino)

# --- 9. Treinamento com SVM radial e validação cruzada 10-fold ------------
controle <- trainControl(method = "cv", number = 10)  # define CV
grid     <- expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.05, 0.1))

set.seed(123)
modelo_cv <- train(
  target ~ .,
  data      = treino_n,
  method    = "svmRadial",    # SVM com kernel radial
  metric    = "Accuracy",     
  tuneGrid  = grid,          
  trControl = controle        
)

print(modelo_cv)  # mostra resultados de cada combinação & melhor

# --- 10. Avaliação no conjunto de teste ----------------------------------
preds <- predict(modelo_cv, newdata = teste_n)  
conf  <- confusionMatrix(preds, teste_n$target)  
print(conf)                              # exibe matriz de confusão
cat("Taxa de erro CV: ", mean(preds != teste_n$target), "\n")  # erro

# --- 11. Função para normalizar e prever novos pacientes ------------------
normalizar_paciente <- function(pac, mean_v, sd_v) {
  nc <- names(pac)[sapply(pac, is.numeric)]
  pac[nc] <- scale(pac[nc], center = mean_v, scale = sd_v)
  pac
}

# --- 12. Definição de perfis de pacientes --------------------------------
paciente3 <- data.frame(
  age = 62,
  sex = factor("male", levels = levels(treino$sex)),
  cp = factor("typical angina", levels = levels(treino$cp)),
  trestbps = 140,
  chol = 230,
  fbs = factor("true", levels = levels(treino$fbs)),
  restecg = factor("ST-T abnormality", levels = levels(treino$restecg)),
  thalach = 150,
  exang = factor("yes", levels = levels(treino$exang)),
  oldpeak = 2.3,
  slope = factor("Flat", levels = levels(treino$slope)),
  ca = 1,
  thal = factor("reversible defect", levels = levels(treino$thal))
)

paciente_saudavel <- data.frame(
  age = 45,
  sex = factor("female", levels = levels(treino$sex)),
  cp = factor("non-anginal pain", levels = levels(treino$cp)),
  trestbps = 118,
  chol = 190,
  fbs = factor("false", levels = levels(treino$fbs)),
  restecg = factor("Normal", levels = levels(treino$restecg)),
  thalach = 165,
  exang = factor("no", levels = levels(treino$exang)),
  oldpeak = 0.3,
  slope = factor("Upsloping", levels = levels(treino$slope)),
  ca = 0,
  thal = factor("normal", levels = levels(treino$thal))
)

paciente_borderline <- data.frame(
  age = 55,
  sex = factor("male", levels = levels(treino$sex)),
  cp = factor("atypical angina", levels = levels(treino$cp)),
  trestbps = 130,
  chol = 240,
  fbs = factor("false", levels = levels(treino$fbs)),
  restecg = factor("ST-T abnormality", levels = levels(treino$restecg)),
  thalach = 160,
  exang = factor("no", levels = levels(treino$exang)),
  oldpeak = 1.4,
  slope = factor("Flat", levels = levels(treino$slope)),
  ca = 1,
  thal = factor("normal", levels = levels(treino$thal))
)

paciente_alto_risco <- data.frame(
  age = 68,
  sex = factor("male", levels = levels(treino$sex)),
  cp = factor("asymptomatic", levels = levels(treino$cp)),
  trestbps = 150,
  chol = 260,
  fbs = factor("true", levels = levels(treino$fbs)),
  restecg = factor("Left ventricular hypertrophy", levels = levels(treino$restecg)),
  thalach = 140,
  exang = factor("yes", levels = levels(treino$exang)),
  oldpeak = 3.5,
  slope = factor("Downsloping", levels = levels(treino$slope)),
  ca = 2,
  thal = factor("fixed defect", levels = levels(treino$thal))
)

paciente_novo <- data.frame(
  age = 52,
  sex = factor("female", levels = levels(treino$sex)),
  cp = factor("asymptomatic", levels = levels(treino$cp)),
  trestbps = 125,
  chol = 220,
  fbs = factor("false", levels = levels(treino$fbs)),
  restecg = factor("Normal", levels = levels(treino$restecg)),
  thalach = 158,
  exang = factor("no", levels = levels(treino$exang)),
  oldpeak = 0.6,
  slope = factor("Upsloping", levels = levels(treino$slope)),
  ca = 0,
  thal = factor("normal", levels = levels(treino$thal))
)

# --- 13. Normalizar e prever cada perfil de paciente ----------------------
paciente3_norm          <- normalizar_paciente(paciente3, media_treino, sd_treino)
previsao_p3             <- predict(modelo_cv, newdata = paciente3_norm)
cat("Paciente 3 =>", as.character(previsao_p3), "\n")

paciente_saudavel_norm  <- normalizar_paciente(paciente_saudavel, media_treino, sd_treino)
previsao_saudavel      <- predict(modelo_cv, newdata = paciente_saudavel_norm)
cat("Paciente saudável =>", as.character(previsao_saudavel), "\n")

paciente_borderline_norm<- normalizar_paciente(paciente_borderline, media_treino, sd_treino)
previsao_borderline     <- predict(modelo_cv, newdata = paciente_borderline_norm)
cat("Paciente borderline =>", as.character(previsao_borderline), "\n")

paciente_alto_risco_norm<- normalizar_paciente(paciente_alto_risco, media_treino, sd_treino)
previsao_alto_risco     <- predict(modelo_cv, newdata = paciente_alto_risco_norm)
cat("Paciente alto risco =>", as.character(previsao_alto_risco), "\n")

paciente_novo_norm      <- normalizar_paciente(paciente_novo, media_treino, sd_treino)
previsao_novo           <- predict(modelo_cv, newdata = paciente_novo_norm)
cat("Paciente novo =>", as.character(previsao_novo), "\n")

# --- 14. Exemplo adicional para slide ------------------------------------
paciente2 <- data.frame(
  age = 40,
  sex = factor("female", levels = levels(treino$sex)),
  cp = factor("non-anginal pain", levels = levels(treino$cp)),
  trestbps = 120,
  chol = 180,
  fbs = factor("false", levels = levels(treino$fbs)),
  restecg = factor("Normal", levels = levels(treino$restecg)),
  thalach = 170,
  exang = factor("no", levels = levels(treino$exang)),
  oldpeak = 0.5,
  slope = factor("Upsloping", levels = levels(treino$slope)),
  ca = 0,
  thal = factor("normal", levels = levels(treino$thal))
)

paciente2_norm <- normalizar_paciente(paciente2, media_treino, sd_treino)
previsao_p2     <- predict(modelo_cv, newdata = paciente2_norm)
cat("Paciente 2 =>", as.character(previsao_p2), "\n")










