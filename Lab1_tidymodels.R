### Tidymodels

## Librerias
library(tidyverse)
library(tidymodels)
library(broom)
library(dotwhisker)
library(patchwork)
library(readr)
library(GGally)
library(here)


# Clase -------------------------------------------------------------------


nrc <- read_csv(here("nrc.csv"))

nrc <- nrc %>% 
  select(rank = R.Rankings.5th.Percentile,
         research = Research.Activity.5th.Percentile,
         student = Student.Support.Outcomes.5th.Percentile,
         diversity = Diversity.5th.Percentile)

# rank (respuestae) vs research (predictora)
a1 <- ggplot(nrc, aes(x = research, y = rank)) + 
  geom_point() + 
  geom_smooth(se = FALSE)
# rank vs student
a2 <- ggplot(nrc, aes(x = student, y = rank)) +
  geom_point() +
  geom_smooth(se = FALSE)
# rank vs diversity
a3 <- ggplot(nrc, aes(x = diversity, y = rank)) + 
  geom_point() +
  geom_smooth(se = FALSE) 

# todos en una fila
a1 + a2 + a3

ggpairs(nrc, columns = c(2,3,4))

lm_mod <-
  parsnip::linear_reg() %>% # Paso 1:  Especificamos el tipo de modelo
  parsnip::set_engine("lm") # Paso 2: Especificamos el motor (engine)
lm_mod

show_engines("linear_reg")

# ---Ajusto el modelo
lm_fit <-
  lm_mod %>% # modelo de parnisp
  parsnip::fit(rank ~ research + student + diversity, # Formula
               data = nrc) # Data frame

lm_xy_fit <-
  lm_mod %>%
  fit_xy(
    x = nrc %>% select(research, student, diversity),
    y = nrc %>% select(rank)
  )

lm_fit

lm_fit$fit %>%
  summary()

# alternativamente
lm_fit %>% 
  extract_fit_engine() %>%
  summary()

# --- coeficientes estimados en formato ordenado
broom::tidy(lm_fit) 

# Coeficientes estimados en formato ordenado (NO CARGO LIBERIA!!!!!)
# lm_fit %>%
#   dotwhisker::dwplot()

# Dot-and-whisker plot
# broom::tidy(lm_fit) %>%
#   dotwhisker::dwplot(dot_args = list(size = 2, color = "black"),
#                      whisker_args = list(color = "black"),
#                      vline = geom_vline(xintercept = 0,
#                                         colour = "grey50"))

# extraer residuos y valores ajustados
# Para modelos de parsnip debemos dar datos para hacer predicciones

broom::augment(lm_fit,
               new_data = nrc)

#resumen del ajuste
broom::glance(lm_fit)

#Extraer los residuos y ajustados
nrc_all <- broom::augment(lm_fit, new_data = nrc)

# Ajustado  vs observado
p1 <-
  ggplot(nrc_all, aes(x = .pred, y = rank)) +
  geom_point() +
  labs(title = "Observado vs predicho") +
  theme(aspect.ratio=1)
# residuos vs. ajustado
p2 <-
  ggplot(nrc_all, aes(x = .pred, y = .resid)) +
  geom_point() +
  labs(title = "residuos vs ajustado") +
  theme(aspect.ratio=1)
require(patchwork)
p1 + p2

# ---saco outliers
nrc_all %>% filter(.pred < 20 & rank > 40)

# Observed (actual) vs predicted (fitted)
# Research
p3 <-
  ggplot(nrc_all, aes(x = research, y = .pred)) +
  geom_point() +theme(aspect.ratio=1)
# Student outcomes
p4 <-
  ggplot(nrc_all, aes(x = student, y = .pred)) +
  geom_point() +theme(aspect.ratio=1)
# Diversity
p5 <-
  ggplot(nrc_all, aes(x = diversity, y = .pred)) +
  geom_point() +theme(aspect.ratio=1)
p3 + p4 + p5

# --- Prediciendo nuevos datos
# Generar un nuevo conjunto de datos usamos expand_grid() )
# Mirar https://tidyr.tidyverse.org/reference/expand_grid.html
new_points <- expand_grid(research = c(10, 40, 70),
                          student = c(10, 40, 70),
                          diversity = c(10, 40, 70))


new_points

# Predecimos nuevos conjuntos de datos unsando el modelo corriente

mean_pred <- predict(lm_fit, new_data = new_points)

# Derivamos los intervalos de confianza

conf_int_pred <- predict(lm_fit,
                         new_data = new_points,
                         type = "conf_int")

# Extraemos los residuos y los valores ajustados,
#agregamos  argument en un data frame
new_points <- broom::augment(lm_fit, new_data = new_points)





# Laboratorio -------------------------------------------------------------


## 1)
ggplot() +
  geom_point(data = nrc_all, aes(x = research, y = .pred)) +
  geom_point(data = new_points, aes(x = research, y = .pred),
             colour = "red") + theme(aspect.ratio = 1)

## 2)
a6 <- ggplot(nrc, aes(x = research, y = rank, color = diversity)) +
  geom_point(size = 3) +
  scale_color_gradient2(midpoint = median(nrc$diversity)) +
  theme(aspect.ratio = 1, legend.position = 'bottom')

a7 <- ggplot(nrc, aes(x = research, y = rank, color = student)) +
  geom_point(size = 3) +
  scale_color_gradient2(midpoint = median(nrc$student)) +
  theme(aspect.ratio = 1,legend.position = 'bottom')


a8 <- ggplot(nrc, aes(x = student, y = rank, color = diversity)) +
  geom_point(size = 3) +
  scale_color_gradient2(midpoint = median(nrc$diversity)) +
  theme(aspect.ratio = 1, legend.position = 'bottom')

a6 + a7 + a8

## 3)
lm_fit_int <-
  lm_mod |> # modelo de parnisp
  parsnip::fit(rank ~ research*diversity , # Formula
               data = nrc) # Data frame
lm_fit_int |>
  tidy()

## 4)
set.seed(1899)
train_test_split <- rsample::initial_split(nrc, prop = 2/3)

nrc_train <- rsample::training(train_test_split)
nrc_test <- rsample::testing(train_test_split)

nrc_lm_fit <-
  lm_mod |>
  fit(rank ~ ., data = nrc_train)

tidy(nrc_lm_fit)

glance(nrc_lm_fit)

## 5)
nrc_lm_train_pred <- augment(nrc_lm_fit, nrc_train)
nrc_lm_test_pred <- augment(nrc_lm_fit, nrc_test)

metrics(nrc_lm_test_pred, truth = rank,
        estimate = .pred)

## 6)
p_f <- ggplot(nrc_lm_train_pred) +
  geom_point(aes(x = .pred, y = rank))

p_e <- ggplot(nrc_lm_train_pred) +
  geom_point(aes(x = .pred, y = .resid))

p_h <- ggplot(nrc_lm_train_pred, aes(x = .resid)) +
  geom_histogram(binwidth=2.5, colour="white") +
  geom_density(aes(y=..count..), bw = 2, colour="orange")

p_q <- ggplot(nrc_lm_train_pred, aes(sample = .resid)) +
  stat_qq() +
  stat_qq_line() +
  xlab("theoretical") + ylab("sample")

p_q + p_e + p_h + p_f

## 7)
p1 <- ggplot(nrc_lm_train_pred) +
  geom_point(aes(x = research, y = rank)) +
  geom_point(aes(x = research, y = .pred),
             colour="blue") +theme(aspect.ratio = 1)

p2 <- ggplot(nrc_lm_train_pred) +
  geom_point(aes(x = student, y = rank)) +
  geom_point(aes(x = student, y = .pred),
             colour="blue") +theme(aspect.ratio = 1)

p3 <- ggplot(nrc_lm_train_pred) +
  geom_point(aes(x = diversity, y = rank)) +
  geom_point(aes(x = diversity, y = .pred),
             colour="blue") +theme(aspect.ratio = 1)

p1 + p2 + p3

