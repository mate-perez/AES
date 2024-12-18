## Librerias
library(tidymodels) 
library(schrute) 
library(vip)
library(ISLR2)
set.seed(4912)


# Clase -------------------------------------------------------------------

office_info <- theoffice |> 
  select(season, episode_name, director, writer,
         character, text, imdb_rating)

office_info |> head()

characters <- office_info |>
  count(episode_name, character) |> # Cantidad de lineas por capitulo de cada personaje
  add_count(character, wt = n, name = "character_count") |> #Cantidad de lineas en toda la serie por personaje.
  filter(character_count > 800) |> #Obtenemos los personajes que tengan al menos 800 lineas
  select(-character_count) |>
  pivot_wider(
    names_from = character,
    values_from = n,
    values_fill = list(n = 0)
  ) #Obtenemos una fila sola por capitulo, ponemos 0 si el personaje no tuvo dialogos en el episodio.

creators <- office_info |>
  distinct(episode_name, director, writer) |>
  pivot_longer(director:writer, names_to = "role", values_to = "person") |>
  separate_rows(person, sep = ";") |>
  add_count(person) |>
  mutate(person = case_when(
    n <= 10 ~ 'Guest',
    n > 10 ~ person
  )) |>
  distinct(episode_name, person) |>
  mutate(person_value = 1) |>
  pivot_wider(
    names_from = person,
    values_from = person_value,
    values_fill = list(person_value = 0)
  )

office <- office_info |> 
  distinct(season, episode_name, imdb_rating) |>
  inner_join(characters) |>
  inner_join(creators) |>
  mutate_at("season", as.factor)

office |>
  ggplot(aes(season, imdb_rating, fill = as.factor(season))) +
  geom_boxplot(show.legend = FALSE)

office_split <- initial_split(office, 
                              strata = season,
                              prop = 3/4) 

office_train <- training(office_split)
office_test <- testing(office_split)

office_rec <- recipe(imdb_rating ~ ., data = office_train) |>
  update_role(episode_name, new_role = "ID") |>
  step_dummy(season) |> #codificamos categorias a columnas binarias donde 1 indica que pertenece a esa clase y 0 que no.
  step_normalize(all_numeric(), -all_outcomes()) #Normalizamos los predictores i.e. media = 0, sd = 1.

tune_spec <- linear_reg(penalty = tune(), mixture = 1) |> #Con tune() indicamos parámetro a ser ajustado.
  set_engine("glmnet") 

tune_wf <- workflow() |>
  add_recipe(office_rec) |>
  add_model(tune_spec)

office_cv <- vfold_cv(office_train, v = 5) #5 particiones debido a que son pocos datos. 
lambda_grid <- grid_regular(penalty(c(-10,-1)),  levels = 50) #Definimos la grilla

lasso_grid <- tune_grid( #Realizamos el ajuste, pensar que estamos haciendo 5*50=250 ajustes!
  tune_wf,
  resamples = office_cv,
  grid = lambda_grid
)

lasso_grid |>
  collect_metrics()

lasso_grid |>
  collect_metrics() |>
  ggplot(aes(penalty, mean, color = .metric)) +
  geom_errorbar(aes(
    ymin = mean - std_err,
    ymax = mean + std_err
  ),
  alpha = 0.5
  ) +
  geom_line(size = 1.5) +
  facet_wrap(~.metric, scales = "free", nrow = 2) +
  scale_x_log10() +
  theme(legend.position = "none")

lowest_rmse <- lasso_grid |>
  select_best(metric = "rmse")

final_lasso <- finalize_workflow(tune_wf, lowest_rmse)

# parentesis(bootstrap)
# office_boot <- bootstraps(office_train, strata = season)
# lambda_grid <- grid_regular(penalty(c(-10,-1)), levels = 50) #Definimos la grilla
# 
# lasso_grid <- tune_grid(
#   tune_wf,
#   resamples = office_boot,
#   grid = lambda_grid
# )
# 

final_lasso |>
  fit(office_train) |>
  extract_fit_engine() |>
  vi(lambda = lowest_rmse$penalty) |> #Es muy importante marcar el lambda!
  ggplot(aes(x = Importance, y = reorder(Variable, Importance), fill = Sign)) +
  geom_col() +
  scale_x_continuous(expand = c(0, 0)) +
  labs(y = NULL)

last_fit(
  final_lasso,
  office_split
) |>
  collect_metrics()



# Laboratorio -------------------------------------------------------------

## 1)
college_split <- initial_split(College, 
                              prop = 4/5) 

college_train <- training(college_split)
college_test <- testing(college_split)

## 2)
lm_mod <-
  parsnip::linear_reg() %>% 
  parsnip::set_engine("lm") 

lm_fit <-
  lm_mod %>% 
  parsnip::fit(Apps ~ ., 
               data = college_train) 

broom::augment(lm_fit,
               new_data = college_train) %>% rmse(Apps, .pred)
###
college_rec <- recipe(Apps ~ ., data = College) %>% 
  step_dummy(Private) %>%  
  step_normalize(all_numeric(), -all_outcomes())

tune_spec <- linear_reg() |> 
  set_engine("lm") 

tune_wf <- workflow() |>
  add_recipe(college_rec) |>
  add_model(tune_spec)

fit1 <- tune_wf %>% parsnip::fit(data = college_train)

broom::augment(fit1,
               new_data = college_train) %>% rmse(Apps, .pred)

## 3)
ridge_mod <-
  linear_reg(penalty = 0, mixture = 0) %>% 
  set_engine("glmnet") 

ridge_fit <-
  ridge_mod %>% 
  fit(Apps ~ ., 
      data = college_train)

bind_rows(
  training = augment(ridge_fit, new_data = college_train) %>%
    rmse(Apps, .pred),
  testing = augment(ridge_fit, new_data = college_test) %>%
    rmse(Apps, .pred)
)

## 4)
ridge_fit %>%
  autoplot()

tidy(ridge_fit, penalty = 1000)

## 5)
college_rec2 <- recipe(Apps ~ ., data = college_train) %>% 
  step_dummy(all_nominal_predictors()) %>%  
  step_normalize(all_predictors())

## 6)
ridge_mod <- linear_reg(penalty = tune(), mixture = 0) %>% 
  set_engine("glmnet")

ridge_wf <- workflow() |>
  add_recipe(college_rec2) |>
  add_model(ridge_mod)

ridge_grid <- grid_regular(penalty(range = c(-5, 3)), levels = 50)

college_cv <- vfold_cv(college_train, v = 5) 

ridge_tune <- tune_grid( 
  ridge_wf,
  resamples = college_cv,
  grid = ridge_grid
)

ridge_tune |>
  collect_metrics()

ridge_tune %>% autoplot()

final_ridge <- finalize_workflow(
  ridge_wf,
  select_best(ridge_tune, metric = "rmse")
)

ridge_last_fit <- last_fit(
  object = final_ridge,
  split = college_split
)

ridge_last_fit %>% 
  collect_metrics()

## 7)
lasso_spec <- linear_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

lasso_wf <- workflow() |>
  add_recipe(college_rec2) |>
  add_model(lasso_spec)

lasso_grid <- grid_regular(penalty(range = c(-5, 3)), levels = 50)

lasso_tune <- tune_grid( 
  lasso_wf,
  resamples = college_cv,
  grid = lasso_grid
)


final_lasso <- finalize_workflow(
  lasso_wf,
  select_best(lasso_tune, metric = "rmse")
)

lasso_last_fit <- last_fit(
  object = final_lasso,
  split = college_split
)

lasso_last_fit %>% 
  collect_metrics()

## 8)
knn_spec <- nearest_neighbor(neighbors = tune()) %>% 
  set_engine("kknn") %>% 
  set_mode("regression")

knn_wf <- workflow() |>
  add_recipe(college_rec2) |>
  add_model(knn_spec)

knn_grid <- grid_regular(neighbors(range = c(1, 50)), levels = 50)

knn_tune <- tune_grid( 
  knn_wf,
  resamples = college_cv,
  grid = knn_grid
)

knn_final <- finalize_workflow(
  knn_workflow,
  select_best(knn_tune, metric = "rmse")
)

knn_last_fit <- last_fit(
  knn_final, split = college_split
)

knn_last_fit |> collect_metrics()




