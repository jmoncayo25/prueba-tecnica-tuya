# Prueba técnica: literal 1
# 19 de marzo de 2023
# Jorge Humberto Moncayo Bravo

setwd("~/Documentos/Pruebas/pruebaTuya/literal_2/src")

# Carga inicial de librerías
library(tidyverse)
library(ggplot2)
library(tidymodels)
library(readxl)
library(tibble)
library(naniar)
library(missForest)
library(polycor)

# Definimos cores para ejecución de modelos en paralelo
cores <- parallel::detectCores()

# Se importa el archivo en formato .xlsx a un data frame en formato tibble
df <- read_excel("../input/ModelosCompetencia.xlsx", 
                                       col_types = c("skip", "skip", "numeric", 
                                                     "text", "numeric")) %>% 
  as_tibble()

colnames(df) <- c("puntaje_AB", "puntaje_XY", "default")
df$puntaje_XY[df$puntaje_XY == "."] <- NA 
df$puntaje_XY <- as.integer(df$puntaje_XY)
df$puntaje_AB <- as.integer(df$puntaje_AB)
df$default <- as.factor(df$default)
df <- na.omit(df)

summary(df) # Se tienen 12800 registros de NA 
sum(is.na(df$puntaje_XY))/nrow(df)*100 # el 11.21% de los datos son nulos

# Se comprueba ubicación de NANs dentro de default y puntaje_XY
ggplot(data = df,
       aes(x = default,
           y = puntaje_XY)) +
  geom_miss_point()

# Se realiza una prueba MCAR de Little (Missing Completely at Random)
## Hipótesis nula es que los datos faltantes son un MCAR
## Valores de p < 0.05 significa que datos faltantes no es un MCAR
## Cuando datos son MCAR -> No hay patrón en los datos faltantes
mcar_test(df[c("puntaje_XY", "default")])

# Evaluación y estabilidad de modelos empleando remuestreo
## Separamos conjuntos de datos
df_AB <- df[c("puntaje_AB", "default")]
df_XY <- df[c("puntaje_XY", "default")]

# Eliminamos NAs de df_XY pues no tienen nivel de provisión asignado
#df_XY <- na.omit(df_XY)

################## Verificación de modelo AB #############################
set.seed(123)
AB_split <- initial_split(df_XY, prop = 0.8, strata = default)
AB_train <- training(AB_split)
AB_test <- testing(AB_split)

lr_mod <- 
  logistic_reg(mode = "classification") %>% 
  set_engine("glm")

set.seed(456)
lr_fit <- 
  lr_mod %>% 
  fit(default ~ ., data = AB_train)
lr_fit

lr_training_pred <- 
  predict(lr_fit, AB_train) %>% 
  bind_cols(predict(lr_fit, AB_train, type = "prob")) %>% 
  # Add the true outcome data back in
  bind_cols(AB_train %>% 
              dplyr::select(default))

lr_training_pred %>%
  accuracy(truth = default, .pred_class)

lr_training_pred %>%
  roc_auc(truth = default, .pred_0)

set.seed(789)
folds <- vfold_cv(AB_train, v = 5)
lr_wf <- 
  workflow() %>%
  add_model(lr_mod) %>%
  add_formula(default ~ .)

set.seed(321)
lr_fit_rs <- 
  lr_wf %>% 
  fit_resamples(folds)

collect_metrics(lr_fit_rs)

pred_test <- lr_fit %>% predict(AB_test) %>% bind_cols(AB_test)
conf_mat(pred_test, default, .pred_class)




biserial.cor(df$puntaje_XY, df$default)
