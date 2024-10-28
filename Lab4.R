## Laboratorio 4

# Librerias
library(tidymodels)
library(ISLR2)
library(discrim)
# library(MASS)
library(kknn)


# Clase -------------------------------------------------------------------

## 1)
auto <- Auto |>
  mutate(alto_consumo = ifelse(mpg > median(mpg), 1, 0)) |>
  mutate(alto_consumo = factor(alto_consumo, levels=c(1,0)), 
         #Importante: El primero es el que toma tidymodel por defecto como clase de interes.
         cylinders = factor(cylinders),
         origin = factor(origin, levels = 1:3, 
                         labels = c("Americano", "Europeo", "Japones"))
  ) #Transformamos a categóricos, para origin agregamos los nombres para mejorar lectura

## 2)

#Para variables numéricas usamos un boxplot.
auto |>
  select(-name, -origin, -mpg, -cylinders) |>
  pivot_longer(-alto_consumo, names_to = "var", values_to = "val") |>
  ggplot(aes(y = alto_consumo, x = val)) +
  geom_boxplot(aes(fill = factor(alto_consumo))) +
  facet_wrap(~var, scales = "free_x") +
  theme(legend.position = "none")  +
  xlab("valor de la variable")

#Para variables categoricas un barplot
auto |>
  select(origin, cylinders, alto_consumo) |>
  pivot_longer(-alto_consumo, names_to = "var", values_to = "val") |>
  ggplot(aes(x = val, fill = alto_consumo)) +
  geom_bar(position = position_dodge()) +
  facet_wrap(~var, scales = "free_x") +
  theme(legend.position = "none")  +
  xlab("valor de la variable") +
  labs(y = "Cantidad")

auto <- auto |> 
  select(-c(mpg, name))

## 3)
set.seed(1234)
auto_split <- initial_split(auto, prop = 3 / 4, strata = alto_consumo) 
#Conservamos proporcion en variable salida
auto_train <- training(auto_split)
auto_test <- testing(auto_split)

## 4)
auto_recipe <- recipe(
  alto_consumo ~ .,
  data = auto_train
) |>
  step_dummy(origin, cylinders)

#Especificamos modelo
lda_spec <- discrim_linear(
  mode = "classification" #Es el unico posible
) |> 
  set_engine('MASS')

#Definimos workflow
lda_workflow <- workflow() |>
  add_recipe(auto_recipe) |>
  add_model(lda_spec)

#Ajustamos modelo
lda_model <- lda_workflow |>
  fit(auto_train)

## 5)
#Especificamos modelo
reglog_spec <- logistic_reg() |> 
  set_engine('glm')

#Definimos workflow
reglog_workflow <- workflow() |>
  add_recipe(auto_recipe) |>
  add_model(reglog_spec)

#Ajustamos modelo
reglog_model <- reglog_workflow |>
  fit(auto_train)

## 6)
auto_recipe <- auto_recipe |> 
  step_normalize(all_numeric_predictors()) #Es necesario en KNN

knn_spec <- nearest_neighbor(
  mode = "classification",
  neighbors = tune()  #Con tune() indicamos parámetro a ser ajustado.
) |> 
  set_engine('kknn')

#Definimos workflow
knn_workflow <- workflow() |>
  add_recipe(auto_recipe) |>
  add_model(knn_spec)

knn_cv <- vfold_cv(auto_train, v = 5) #Hacemos validación cruzada con 5 particiones
knn_grid <- grid_regular(neighbors(c(2, 17)), levels=15) #Definimos la grilla. 17 = sqrt(nrow(auto))

knn_tune <- tune_grid( #Realizamos el ajuste, pensar que estamos haciendo 5*17=85 
  knn_workflow,
  resamples = knn_cv,
  grid = knn_grid
)

lowest_auc <- knn_tune |>
  select_best(metric = "roc_auc") 
#Usamos AUC como medida para comparar los modelos. El mejor fue k=12

knn_last_workflow <- finalize_workflow(knn_workflow, lowest_auc)
knn_model <- knn_last_workflow |>
  fit(auto_train)

## 7)
auto_metrics <- metric_set(accuracy, sens, spec) #Definimos las metricas.
models <- list("lda" = lda_model, "reglog" = reglog_model, "knn" = knn_model)

results <- lapply(models, function(model) { 
  #Para cada modelo calculamos las metricas.
  model |>
    augment(new_data = auto_test) |> #Realizamos prediccion
    auto_metrics(predictions, truth = alto_consumo, estimate = .pred_class) 
  #Calculo de metricas definidas
})

results <- list_rbind(results, names_to = "model") #Juntamos resultados

results |> 
  select(-c(.estimator)) |>
  pivot_wider(names_from = .metric, values_from = .estimate)

mat_conf <- lapply(models, function(model) { 
  #Para cada modelo calculamos la matriz de confusion
  model |>
    augment(new_data = auto_test) |>
    count(alto_consumo, .pred_class) |> 
    #Realizamos el conteo de las distintas combinaciones obs-pred
    mutate(.pred_class = factor(.pred_class,c(0,1))) 
  #Solo para mostrar el eje y con el 1 arriba
})

mat_conf <- list_rbind(mat_conf, names_to = "model")

mat_conf |> 
  ggplot(aes(x = alto_consumo, y = .pred_class)) +
  geom_tile(aes(fill = n)) +
  geom_text(aes(label = n), color = "white") +
  labs(x = "Observado", y = "Predicho") +
  theme(legend.position = "none") +
  facet_wrap(~ model, ncol = 2) 

curves <- lapply(models, function(model) {
  model |>
    augment(new_data = auto_test) |>
    roc_curve(truth = alto_consumo, .pred_1) #Calculamos la curva ROC.
})

curves <- list_rbind(curves, names_to = "model")

ggplot(curves, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path() +
  coord_equal() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(x = "1 - Specificity",
       y = "Sensitivity",
       color = "Modelo") 


# Laboratorio -------------------------------------------------------------


