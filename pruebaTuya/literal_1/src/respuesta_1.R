# Prueba técnica: literal 1
# 18 de marzo de 2023
# Jorge Humberto Moncayo Bravo

setwd("~/Documentos/Pruebas/pruebaTuya/literal_1/src")

# Carga inicial de librerías
library(tidyverse)
library(ggplot2)
library(tidymodels)
library(readxl)
library(tibble)
library(ranger)
library(randomForest)

# Se importa el archivo en formato .xlsx a un data frame en formato tibble
df <- read_excel("../input/ProductoNuevo.xlsx", 
                            col_types = c("skip", "skip", "numeric", 
                                          "numeric", "numeric", "numeric", 
                                          "numeric", "numeric", "numeric", 
                                          "numeric", "text", "text", "numeric", 
                                          "text", "text", "numeric", "text", 
                                          "text", "numeric", "numeric", "numeric", 
                                          "numeric")) %>% 
  as_tibble()

# Se modifican nombres de columnas
colnames(df) <- c("mora30", "mora60", "moramax_ultimosemestre", 
                  "experiencia", "personas_cargo", "gastos_familiares",
                  "gastos_arriendo", "tiempo_actividad", "ocupacion",
                  "tipo_contrato", "edad", "estado_civil",
                  "genero", "ingresos", "nivel_academico", 
                  "tipo_vivienda", "tiempo_cliente","tiempo_sistema", 
                  "porcend", "obligaciones_sistema")

# Se cambia el tipo de datos de las columnas en el dataframe
df <- df %>% 
  mutate(mora30 = as.factor(mora30),
         mora60 = as.factor(mora60),
         moramax_ultimosemestre = as.integer(moramax_ultimosemestre),
         experiencia = as.logical(experiencia),
         personas_cargo = as.integer(personas_cargo),
         gastos_familiares = as.integer(gastos_familiares),
         gastos_arriendo = as.integer(gastos_arriendo),
         edad = as.integer(edad),
         ingresos = as.integer(ingresos),
         tiempo_cliente = as.integer(tiempo_cliente),
         tiempo_sistema = as.integer(tiempo_sistema),
         obligaciones_sistema = as.integer(obligaciones_sistema))
         
# Se visualiza en consola de R el resumen estadístico de variables numéricas
summary(df)

# Por inspección visual se detecta outlier en tiempo_actividad (TiempoActividad)

## Se calcula mediana de columna tiempo_actividad
mediana <- median(df$tiempo_actividad, na.rm = TRUE)

## Se reemplaza outlier por la mediana
outlier <- 866880.0
df$tiempo_actividad[df$tiempo_actividad == outlier] <- mediana

# Se hace summary de columna para verificar cambio
summary(df$tiempo_actividad)

# Creación de modelo predictivo usando tidymodels

df <- select(df, -mora60) # Eliminación de mora60

## Creación de conjuntos de prueba y test
set.seed(1)
df_split <- initial_split(df, prop = 0.8, strata = mora30) # Proporción debida a heurísticas, estratificación

## Se crea receta dentro del flujo de trabajo de tidymodels
df_rec <- training(df_split) %>% 
  recipe(mora30 ~ .) %>% 
  # Se aplica codificación one hot sobre variables categóricas
  step_dummy(all_nominal_predictors()) %>% 
  # Eliminar variables redundantes que están altamente correlacionadas
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  # Se aplica PCA para disminuir redundancia
  step_pca(all_numeric_predictors()) %>%
  # Normalización de las variables numéricas
  step_normalize(all_numeric_predictors()) %>% 
  # Escalamiento de las variables numéricas
  step_scale(all_numeric_predictors()) %>% 
  prep()

df_training <- juice(df_rec)
df_testing <- df_rec %>% 
  bake(testing(df_split))

model_rf <- rand_forest(trees = 100) %>% 
  set_engine("randomForest") %>% 
  set_mode("classification")

rf_fit <- 
  model_rf %>% 
  fit(mora30 ~ ., data = df_training)

rf_training_pred <- 
  predict(rf_fit, df_training) %>% 
  bind_cols(predict(rf_fit, df_training, type = "prob")) %>% 
  # Add the true outcome data back in
  bind_cols(df_training %>% 
              select(mora30))

rf_training_pred %>%
  roc_auc(truth = mora30, .pred_1)

rf_training_pred %>%
  accuracy(truth = mora30, .pred_class)

rf_testing_pred <- 
  predict(rf_fit, df_testing) %>% 
  bind_cols(predict(rf_fit, df_testing, type = "prob")) %>% 
  bind_cols(df_testing %>% select(mora30))

rf_testing_pred %>%                 
  roc_auc(truth = mora30, .pred_1)

rf_testing_pred %>%                 
  accuracy(truth = mora30, .pred_class)

folds <- vfold_cv(df_training, v = 5)

rf_wf <- 
  workflow() %>%
  add_model(model_rf) %>%
  add_formula(mora30 ~ .)

rf_fit_rs <- 
  rf_wf %>% 
  fit_resamples(folds)

rf_summary <- rf_fit_rs %>% 
  collect_metrics()


best_model <- select_best(rf_fit_rs, "roc_auc")

predictiones <- predict(best_model, df_testing)







collect_metrics(rf_fit_rs)

collect_predictions(rf_fit_rs)











df_rf <- rand_forest(trees = 100, mode = "classification") %>% 
  set_engine("randomForest") %>% 
  fit(mora30 ~ ., data = df_training)

df_rf %>%
  predict(df_testing) %>%
  bind_cols(df_testing) %>% 
  metrics(truth = mora30, estimate = .pred_class)
