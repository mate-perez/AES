### Laboratorio 5 - arboles


# Clase -------------------------------------------------------------------

library(tidymodels)
library(ISLR)
library(rpart.plot)
library(vip)

data("Boston", package = "MASS")

Boston <- as_tibble(Boston)

Carseats <- as_tibble(Carseats) %>%
  mutate(High = factor(if_else(Sales <= 8, "No", "Yes"))) %>%
  select(-Sales)

tree_spec <- decision_tree() %>%
  set_engine("rpart")

class_tree_spec <- tree_spec %>%
  set_mode("classification")

class_tree_fit <- class_tree_spec %>%
  fit(High ~ ., data = Carseats)

class_tree_fit

class_tree_fit %>%
  extract_fit_engine() %>%
  rpart.plot(roundint=FALSE)

augment(class_tree_fit, new_data = Carseats) %>%
  accuracy(truth = High, estimate = .pred_class)

augment(class_tree_fit, new_data = Carseats) %>%
  conf_mat(truth = High, estimate = .pred_class)

set.seed(1234)
Carseats_split <- initial_split(Carseats)

Carseats_train <- training(Carseats_split)
Carseats_test <- testing(Carseats_split)

class_tree_fit <- fit(class_tree_spec, High ~ ., 
                      data = Carseats_train)

augment(class_tree_fit, new_data = Carseats_train) %>%
  conf_mat(truth = High, estimate = .pred_class)

augment(class_tree_fit, new_data = Carseats_test) %>%
  conf_mat(truth = High, estimate = .pred_class)

augment(class_tree_fit, new_data = Carseats_test) %>%
  accuracy(truth = High, estimate = .pred_class)

class_tree_wf <- workflow() %>%
  add_model(class_tree_spec %>% 
              set_args(cost_complexity = tune())) %>%
  add_formula(High ~ .)

set.seed(1234)
Carseats_fold <- vfold_cv(Carseats_train) #defecto 10-folds

param_grid <- grid_regular(cost_complexity(range = c(0.001, 0.15), trans=NULL), 
                           levels = 10)

tune_res <- tune_grid(
  class_tree_wf, 
  resamples = Carseats_fold, 
  grid = param_grid, 
  metrics = metric_set(accuracy)
)

autoplot(tune_res)

best_complexity <- select_best(tune_res)

class_tree_final <- finalize_workflow(class_tree_wf, best_complexity)

class_tree_final_fit <- fit(class_tree_final, data = Carseats_train)

class_tree_final_fit

class_tree_final_fit %>%
  extract_fit_engine() %>% #extrae el objeto rpart
  rpart.plot(roundint=FALSE)

reg_tree_spec <- tree_spec %>%
  set_mode("regression")

set.seed(1234)
Boston_split <- initial_split(Boston)

Boston_train <- training(Boston_split)
Boston_test <- testing(Boston_split)

reg_tree_fit <- fit(reg_tree_spec, medv ~ ., Boston_train)
reg_tree_fit

augment(reg_tree_fit, new_data = Boston_test) %>%
  rmse(truth = medv, estimate = .pred)

reg_tree_fit %>%
  extract_fit_engine() %>%
  rpart.plot(roundint=FALSE)

reg_tree_wf <- workflow() %>%
  add_model(reg_tree_spec %>% set_args(cost_complexity = tune())) %>%
  add_formula(medv ~ .)

set.seed(1234)
Boston_fold <- vfold_cv(Boston_train)

param_grid <- grid_regular(cost_complexity(range = c(0, 0.1), trans=NULL),
                           levels = 20)

tune_res <- tune_grid(
  reg_tree_wf, 
  resamples = Boston_fold, 
  grid = param_grid
)

autoplot(tune_res)

best_complexity <- select_best(tune_res, metric = "rmse")

reg_tree_final <- finalize_workflow(reg_tree_wf, best_complexity)

reg_tree_final_fit <- fit(reg_tree_final, data = Boston_train)
reg_tree_final_fit

reg_tree_final_fit %>%
  extract_fit_engine() %>%
  rpart.plot(roundint=FALSE,)

library(randomForest)

bagging_spec <- rand_forest(mtry = .cols()) %>%
  set_engine("randomForest", importance = TRUE) %>%
  set_mode("classification")

bagging_fit <- fit(bagging_spec, High ~ ., data = Carseats_train)

augment(bagging_fit, new_data = Carseats_test) %>%
  accuracy(truth = High, estimate = .pred_class)

vip(bagging_fit)

rf_spec <- rand_forest(mtry = 6) %>%
  set_engine("randomForest", importance = TRUE) %>%
  set_mode("classification")
rf_fit <- fit(rf_spec, High ~ ., data = Carseats_train)
rf_fit 

rf_training_pred <- 
  predict(rf_fit, Carseats_train) %>% 
  bind_cols(predict(rf_fit, Carseats_train, type = "prob")) %>% 
  # Agragamos los verdaderos datosd
  bind_cols(Carseats_train %>% 
              select(High))

rf_training_pred %>%                # training set predictions
  roc_auc(truth = High, .pred_No)

rf_training_pred %>%               
  accuracy(truth = High, .pred_class)

rf_test_pred <- 
  predict(rf_fit, Carseats_test) %>% 
  bind_cols(predict(rf_fit, Carseats_test, type = "prob")) %>% 
  # Agragamos los verdaderos datosd
  bind_cols(Carseats_test %>% 
              select(High))

rf_test_pred %>%               
  roc_auc(truth = High, .pred_No)

autoplot(roc_curve(rf_test_pred, truth = High, .pred_No))

augment(rf_fit, new_data = Carseats_test) %>%
  accuracy(truth = High, estimate = .pred_class)

vip(rf_fit)


# Laboratorio -------------------------------------------------------------

## 1)
library(ranger)
tune_spec <- rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")
tune_wf <- workflow() %>%
  add_model(tune_spec %>% 
              set_args(mtry = tune(),
                       min_n = tune())) %>%
  add_formula(High ~ .)
tune_fit <- fit(tune_spec, High ~ ., data = Carseats_train)
tune_fit 

## 2) y 3)
set.seed(321)
trees_fold <- vfold_cv(Carseats_train) 

tune_res <- tune_grid(tune_wf,
                      resamples = trees_fold,
                      grid = 20,
                      metrics = metric_set(roc_auc))

## 4)
autoplot(tune_res)

## 5)
rf_grid <- grid_regular(
  mtry(range = c(2, 7)),
  min_n(range = c(2, 10)),
  levels = 5
)

set.seed(456)

regular_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = rf_grid
)
regular_res

regular_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  mutate(min_n = factor(min_n)) %>%
  ggplot(aes(mtry, mean, color = min_n)) +
  geom_line(alpha = 0.5, size = 1.5) +
  geom_point() +
  labs(y = "AUC") 

## 6)
best_auc <- select_best(regular_res, metric="roc_auc")

final_rf <- finalize_model(
  tune_spec,
  best_auc
)
final_rf

final_wf <- workflow() %>%
  add_recipe(rf_ranger) %>%
  add_model(final_rf)

final_res <- final_wf %>%
  last_fit(Carseats_split)

final_res %>%
  collect_metrics()