# Previsão de Doença Cardíaca com SVM em R

Este repositório contém um projeto em R que utiliza **Support Vector Machines (SVM)** para prever a presença de doenças cardíacas a partir de dados clínicos.

---

## 1. Pacotes Necessários

O projeto utiliza os seguintes pacotes:

- `readr` → leitura de arquivos CSV
- `dplyr` → manipulação de data.frames
- `e1071` → implementação de SVM
- `caret` → treino e validação de modelos

> O código verifica se os pacotes estão instalados e instala automaticamente caso necessário.

```r
if (!require("readr")) install.packages("readr")
if (!require("dplyr")) install.packages("dplyr")
if (!require("e1071")) install.packages("e1071")
if (!require("caret")) install.packages("caret")

library(readr)
library(dplyr)
library(e1071)
library(caret)
```r
