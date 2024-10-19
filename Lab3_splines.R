### Laboratorio 3


# Clase -------------------------------------------------------------------

library(tidymodels)
library(ISLR)
library(workflows)
library(parsnip)
library(rsample)

Wage <- as_tibble(Wage)
glimpse(Wage)

set.seed(123)

data_split <- rsample::initial_split(Wage, prop = 3/4)

# Create data frames for the two sets:
train_data <- rsample::training(data_split)
test_data  <- rsample::testing(data_split)

rec_poly <- recipes::recipe(wage ~ age, data = train_data) %>%
  recipes::step_poly(age, degree = 4, options = list(raw = TRUE))

lm_spec <- parsnip::linear_reg() %>%
  parsnip::set_mode("regression") %>% 
  parsnip::set_engine("lm")

poly_wf <- workflows::workflow() %>%
  workflows::add_model(lm_spec) %>%
  workflows::add_recipe(rec_poly)

poly_fit <- parsnip:: fit(poly_wf, data = train_data)
poly_fit
tidy(poly_fit)

age_range <- tibble(age = seq(min(Wage$age), max(Wage$age)))

regression_lines <- bind_cols(
  augment(poly_fit, new_data = age_range),
  predict(poly_fit, new_data = age_range, type = "conf_int")
)
regression_lines

Wage %>%
  ggplot(aes(age, wage)) +
  geom_point(alpha = 0.2) +
  geom_line(aes(y = .pred), color = "darkgreen",
            data = regression_lines) +
  geom_line(aes(y = .pred_lower), data = regression_lines,
            linetype = "dashed", color = "blue") +
  geom_line(aes(y = .pred_upper), data = regression_lines,
            linetype = "dashed", color = "blue")


# Laboratorio -------------------------------------------------------------

## Parte 1

### 1)
lm_grad4 <- lm(wage ~ poly(age, degree = 4, raw = T), data = train_data)

### 2)
summary(lm_grad4)
tidy(poly_fit)

### 3)
# No debería dar diferente si se usa raw = T y los mismos datos

### 4)
data <- train_data %>% 
  mutate(age2 = age**2,
         age3 = age**3,
         age4 = age**4)

lm_grad4_2 <- lm(wage ~ age + age2 + age3 + age4, data = data)

summary(lm_grad4_2)


## Parte 2

### 1)
rec_poly_2 <- recipes::recipe(wage ~ age, data = train_data) %>%
  recipes::step_poly(age, degree = 4)

### 2)
poly_wf_2 <- workflows::workflow() %>%
  workflows::add_model(lm_spec) %>%
  workflows::add_recipe(rec_poly_2)

poly_fit_2 <- parsnip:: fit(poly_wf_2, data = train_data)
tidy(poly_fit_2)

### 3)
# NO

### 4)
regression_lines_2 <- bind_cols(
  augment(poly_fit_2, new_data = age_range),
  predict(poly_fit_2, new_data = age_range, type = "conf_int")
)
regression_lines_2

Wage %>%
  ggplot(aes(age, wage)) +
  geom_point(alpha = 0.2) +
  geom_line(aes(y = .pred), color = "darkgreen",
            data = regression_lines_2) +
  geom_line(aes(y = .pred_lower), data = regression_lines_2,
            linetype = "dashed", color = "blue") +
  geom_line(aes(y = .pred_upper), data = regression_lines_2,
            linetype = "dashed", color = "blue")


## Parte 3

### 1)
rec_spline <- recipes::recipe(wage ~ age, data = train_data) %>%
  step_bs(age, options = list(knots = 25, 40, 60))

### 2)
spline_wf <- workflows::workflow() %>%
  workflows::add_model(lm_spec) %>%
  workflows::add_recipe(rec_spline)

### 3)
spline_fit <- parsnip:: fit(spline_wf, data = train_data)
tidy(spline_fit)

### 4)
augment(spline_fit,
        new_data = test_data)

### 5)
age_range <- tibble(age = seq(min(Wage$age), max(Wage$age)))

regression_lines_3 <- bind_cols(
  augment(spline_fit, new_data = age_range),
  predict(spline_fit, new_data = age_range, type = "conf_int")
)
regression_lines_3

### 6)
Wage %>%
  ggplot(aes(age, wage)) +
  geom_point(alpha = 0.2) +
  geom_line(aes(y = .pred), color = "darkgreen",
            data = regression_lines_3) +
  geom_line(aes(y = .pred_lower), data = regression_lines_3,
            linetype = "dashed", color = "blue") +
  geom_line(aes(y = .pred_upper), data = regression_lines_3,
            linetype = "dashed", color = "blue")


### Parte 4

rec_spline2 <- recipes::recipe(wage ~ age, data = train_data) %>%
  step_ns(age, options = list(knots = 25, 40, 60))

spline_wf2 <- workflows::workflow() %>%
  workflows::add_model(lm_spec) %>%
  workflows::add_recipe(rec_spline2)

spline_fit2 <- parsnip:: fit(spline_wf2, data = train_data)
tidy(spline_fit2)

regression_lines_4 <- bind_cols(
  augment(spline_fit2, new_data = age_range),
  predict(spline_fit2, new_data = age_range, type = "conf_int")
)
regression_lines_4

Wage %>%
  ggplot(aes(age, wage)) +
  geom_point(alpha = 0.2) +
  geom_line(aes(y = .pred), color = "darkgreen",
            data = regression_lines_3) +
  geom_line(aes(y = .pred_lower), data = regression_lines_3,
            linetype = "dashed", color = "blue") +
  geom_line(aes(y = .pred_upper), data = regression_lines_3,
            linetype = "dashed", color = "blue") +
  geom_line(aes(y = .pred), color = "red",
            data = regression_lines_4) +
  geom_line(aes(y = .pred_lower), data = regression_lines_4,
            linetype = "dashed", color = "orange") +
  geom_line(aes(y = .pred_upper), data = regression_lines_4,
            linetype = "dashed", color = "orange")


## Parte 5

rec_gam <- recipes::recipe(wage ~ age + year + education, data = train_data) %>%
  step_ns(age, deg_free = 5) %>% 
  step_ns(year, deg_free = 4) 

gam_wf <- workflows::workflow() %>%
  workflows::add_model(lm_spec) %>%
  workflows::add_recipe(rec_gam)

gam_fit <- parsnip:: fit(gam_wf, data = train_data)
tidy(gam_fit)

augment(gam_fit,
        new_data = test_data)

library(gratia)
library(mgcv)

rec_gam2 <- recipes::recipe(wage ~ age + year + education, data = Wage) 

mgcv_spec <- parsnip::gen_additive_mod() %>%
  parsnip::set_engine("mgcv") %>% 
  parsnip::set_mode("regression")   

gam_wf <- workflows::workflow() %>%
  workflows::add_recipe(rec_gam2) %>% 
  workflows::add_model(mgcv_spec, formula = wage ~ s(age, k=5) + s(year, k=4) + education)

gam_fit <- parsnip::fit(gam_wf, data = train_data)
gam <- extract_fit_engine(gam_fit)

gratia::draw(gam, residuals = T)
