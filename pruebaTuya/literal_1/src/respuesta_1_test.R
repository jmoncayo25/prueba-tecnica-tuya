setwd("~/Documentos/Pruebas/pruebaTuya/literal_1/src")

# Carga inicial de librerías
library(tidyverse)
library(ggplot2)
library(tidymodels)
library(readxl)
library(tibble)
library(parsnip)
library(readr)
library(vip)

cores <- parallel::detectCores()

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
         experiencia = as.integer(experiencia),
         personas_cargo = as.integer(personas_cargo),
         gastos_familiares = as.integer(gastos_familiares),
         gastos_arriendo = as.integer(gastos_arriendo),
         edad = as.integer(edad),
         ingresos = as.integer(ingresos),
         tiempo_cliente = as.integer(tiempo_cliente),
         tiempo_sistema = as.integer(tiempo_sistema),
         obligaciones_sistema = as.integer(obligaciones_sistema))

mediana <- median(df$tiempo_actividad, na.rm = TRUE)
outlier <- 866880.0
df$tiempo_actividad[df$tiempo_actividad == outlier] <- mediana
df <- df %>% select(-mora60, -moramax_ultimosemestre)
set.seed(1)
splits <- initial_split(df, strata = mora30)
df_training <- training(splits)
df_testing <- testing(splits)
set.seed(2)
df_validation <- validation_split(df_training, 
                            strata = mora30, 
                            prop = 0.80)

model_rf <- rand_forest(trees = tune(), mtry = tune(), min_n = tune()) %>% 
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("classification")

model_lr <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet", num.threads = cores)

model_dt <- decision_tree(tree_depth = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

model_svm <- svm_rbf(cost = tune(), rbf_sigma = tune()) %>% 
  set_engine("kernlab", num.threads = cores) %>% 
  set_mode("classification")

model_xboost <- boost_tree(trees = tune(), 
                           loss_reduction = tune(),
                           min_n = tune(),
                           learn_rate = tune()) %>% 
  set_engine("xgboost", num.threads = cores) %>% 
  set_mode("classification")

model_neural <- mlp(hidden_units = tune(), 
                    activation = tune(), 
                    dropout = tune(),
                    epochs = 30) %>% 
  set_engine("keras", num.threads = cores) %>%
  set_mode("classification")
  

df_rec <- recipe(mora30 ~ ., data = df_training) %>% 
  # Se aplica codificación one hot sobre variables categóricas
  step_dummy(all_nominal_predictors()) %>% 
  # Eliminar variables redundantes que están altamente correlacionadas
  step_corr(all_numeric_predictors(), threshold = 0.9) %>%
  # Se aplica PCA para disminuir redundancia
  #step_pca(all_numeric_predictors()) %>%
  # Normalización de las variables numéricas
  step_normalize(all_numeric_predictors()) %>% 
  # Escalamiento de las variables numéricas
  step_scale(all_numeric_predictors())

rf_workflow <- 
  workflow() %>% 
  add_model(model_rf) %>% 
  add_recipe(df_rec)

lr_workflow <- 
  workflow() %>% 
  add_model(model_lr) %>% 
  add_recipe(df_rec)

dt_workflow <- 
  workflow() %>% 
  add_model(model_dt) %>% 
  add_recipe(df_rec)

svm_workflow <- 
  workflow() %>% 
  add_model(model_svm) %>% 
  add_recipe(df_rec)

xboost_workflow <- 
  workflow() %>% 
  add_model(model_xboost) %>% 
  add_recipe(df_rec)

neural_workflow <- 
  workflow() %>% 
  add_model(model_neural) %>% 
  add_recipe(df_rec)
  

rf_grid <- tibble(trees = seq(30, 200, by = 5),
                  mtry = runif(35, min = 1, max = 10))
rf_grid <- tibble(trees = seq(50, 100, by = 5),
                  mtry = seq(10, 1, length.out = 11))
grid_lr <- tibble(penalty = 10^seq(-4, -1, length.out = 30))
dt_grid <- tibble(tree_depth = seq(1, 30, by = 1))
svm_grid <- tibble(cost = runif(20, min = 0.1, max = 5),
                   rbf_sigma = runif(20, min = 0.0001, max = 0.001))
xboost_grid <- tibble(min_n = seq(1, 50, length.out = 50))


trees_bounds <- c(500, 1000)
loss_reduction_bounds <- c(0.1, 1)
min_n_bounds <- c(1, 50)
learning_rate_bounds <- c(0.01, 0.3)

# Generar valores aleatorios para cada hiperparámetro
n_combinations <- 30
trees_values <- sample.int(trees_bounds[2] - trees_bounds[1], n_combinations, replace = TRUE) + trees_bounds[1]
loss_reduction_values <- runif(n_combinations, loss_reduction_bounds[1], loss_reduction_bounds[2])
min_n_values <- sample.int(min_n_bounds[2] - min_n_bounds[1], n_combinations, replace = TRUE) + min_n_bounds[1]
learning_rate_values <- runif(n_combinations, learning_rate_bounds[1], learning_rate_bounds[2])

trees_bounds_rf <- c(10, 100)
min_n_bounds_rf <- c(1, 50)
mtry_bounds_rf <- c(1, 10)

# Generar valores aleatorios para cada hiperparámetro
n_combinations <- 100
trees_values_rf <- sample.int(trees_bounds_rf[2] - trees_bounds_rf[1], 
                              n_combinations, replace = TRUE) + trees_bounds_rf[1]
min_n_values_rf <- sample.int(min_n_bounds_rf[2] - min_n_bounds_rf[1], 
                           n_combinations, replace = TRUE) + min_n_bounds_rf[1]
mtry_values_rf <- runif(n_combinations, mtry_bounds_rf[1], mtry_bounds_rf[2])

# Crear una grilla aleatoria en tibble
grid <- tibble(
  trees = trees_values,
  loss_reduction = loss_reduction_values,
  min_n = min_n_values,
  learn_rate = learning_rate_values
)

grid_neural <- crossing(
  hidden_units = c(4, 8, 16, 32, 64, 128),
  activation = c("relu"),
  dropout = c(0.01, 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99),
)

grid_rf <- tibble(
  trees = trees_values_rf,
  min_n = min_n_values_rf,
  mtry = mtry_values_rf
)

rf_res <- 
  rf_workflow %>% 
  tune_grid(df_validation,
            grid = grid_rf,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc, precision, sensitivity))

lr_res <- 
  lr_workflow %>% 
  tune_grid(df_validation,
            grid = grid_lr,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc, precision, sensitivity))

dt_res <- 
  dt_workflow %>% 
  tune_grid(df_validation,
            grid = dt_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

svm_res <- 
  svm_workflow %>% 
  tune_grid(df_validation,
            grid = svm_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

xboost_res <- 
  xboost_workflow %>% 
  tune_grid(df_validation,
            grid = xboost_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

xboost_res <- 
  xboost_workflow %>% 
  tune_grid(df_validation,
            grid = grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc, precision, sensitivity))

neural_res <- 
  neural_workflow %>% 
  tune_grid(df_validation,
            grid = grid_neural,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc, precision, sensitivity))

rf_plot <- 
  rf_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = trees, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

rf_plot <- 
  rf_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = mtry, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

rf_plot

lr_plot <- 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

dt_plot <- 
  dt_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = tree_depth, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

svm_plot <- 
  svm_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = cost, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

xboost_plot <- 
  xboost_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = min_n, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

rf_plot
lr_plot
dt_plot
svm_plot
xboost_plot

rf_best <- 
  rf_res %>% 
  collect_metrics() %>% 
  spread(.metric, mean) %>% 
  arrange(desc(roc_auc))
rf_top <- 
  rf_res %>% 
  select_best(metric = "roc_auc")
rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_top) %>% 
  roc_curve(mora30, .pred_0) %>% 
  mutate(model = "Random Forest")

lr_best <- 
  lr_res %>% 
  collect_metrics() %>% 
  spread(.metric, mean) %>% 
  arrange(desc(roc_auc))
lr_top <- 
  lr_res %>% 
  select_best(metric = "roc_auc")
lr_auc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_top) %>% 
  roc_curve(mora30, .pred_0) %>% 
  mutate(model = "Logistic Regression")

neural_best <- 
  neural_res %>% 
  collect_metrics() %>% 
  spread(.metric, mean) %>% 
  arrange(desc(roc_auc))
neural_top <- 
  neural_res %>% 
  select_best(metric = "roc_auc")
neural_auc <- 
  neural_res %>% 
  collect_predictions(parameters = neural_top) %>% 
  roc_curve(mora30, .pred_0) %>% 
  mutate(model = "Neural Network")


bind_rows(rf_auc, lr_auc, neural_auc) %>% 
  ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
  geom_path(linewidth = 1.5, alpha = 0.8) +
  geom_abline(lty = 3) + 
  coord_equal() + 
  scale_color_viridis_d(option = "plasma", end = .6) +
  labs(title = "Curvas ROC de diferentes modelos",
       x = "1 - Especificidad", y = "Sensibilidad", col = "Modelos") +
  scale_x_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        legend.position = "bottom", # Ubica la leyenda debajo de la gráfica
        legend.justification = "center", # Centra la leyenda
        legend.title = element_text(size = 14),
        axis.title = element_text(size = 14, face = "bold"),
        axis.text = element_text(size = 12))


dt_best <- 
  dt_res %>% 
  collect_metrics() %>% 
  arrange(tree_depth) %>% 
  slice(20)

xboost_best <- 
  xboost_res %>% 
  collect_metrics()

neural_best <- 
  neural_res %>% 
  collect_metrics()

rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(mora30, .pred_0) %>% 
  mutate(model = "Random Forest")

lr_auc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(mora30, .pred_0) %>% 
  mutate(model = "Logistic Regression")

dt_auc <- 
  dt_res %>% 
  collect_predictions(parameters = dt_best) %>% 
  roc_curve(mora30, .pred_0) %>% 
  mutate(model = "Decision Tree")

autoplot(rf_auc)
autoplot(lr_auc)
autoplot(dt_auc)


# the last model
last_rf_mod <- 
  mlp(hidden_units = 4, dropout = 0.010, activation = "relu", epochs = 20) %>% 
  set_engine("keras", num.threads = cores) %>% 
  set_mode("classification")

# the last workflow
last_rf_workflow <- 
  rf_workflow %>% 
  update_model(last_rf_mod)

# the last fit
set.seed(345)
last_rf_fit <- 
  last_rf_workflow %>% 
  last_fit(splits)

last_rf_fit

last_rf_fit %>% 
  collect_metrics()

last_rf_fit %>% 
  extract_fit_parsnip() %>% 
  vip(num_features = 15)
