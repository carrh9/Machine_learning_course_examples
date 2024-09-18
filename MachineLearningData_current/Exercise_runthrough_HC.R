#### Load required R packages ####
library(tidyverse)
library(tidymodels)
tidymodels_prefer()

#### Load input data ####
read_delim("../../MachineLearningData/transmembrane_data.txt") -> data

#### Preparing data for modelling ####
# Turning transmembrane into a factor
data %>%
  mutate(
    transmembrane = factor(transmembrane)
  ) -> data

# Remove gene_id column
data %>%
  select(-gene_id) -> data

# Shuffle the rows
data %>%
  sample_frac() -> data

# Remove missing values
data %>%
  na.omit() -> data

# Split the data
data %>%
  initial_split(prop=0.8, strata=transmembrane) -> split_data

training(split_data)
testing(split_data)

# Build the model
rand_forest(trees=100) %>%
  set_engine("ranger") %>%
  set_mode("classification") -> forest_model

forest_model %>% translate()

# Train the model
forest_model %>%
  fit(transmembrane ~ ., data=training(split_data)) -> forest_fit #predict transmembrane, given everything else

forest_fit

# Test the model
forest_fit %>%
  predict(testing(split_data))

forest_fit %>%
  predict(testing(split_data)) %>%
  bind_cols(testing(split_data)) -> prediction_results

# Evaluate the predictions
prediction_results %>%
  group_by(transmembrane, .pred_class) %>%
  count() #gives counts

prediction_results %>%
  sens(transmembrane, .pred_class) #gives sensitivity
prediction_results %>%
  spec(transmembrane, .pred_class) #gives specificity

prediction_results %>%
  metrics(transmembrane, .pred_class) #gives accuracy and Cohen's kappa value


# Build a different model
rand_forest(trees=100, mtry = 10, min_n = 5) %>%
  set_engine("ranger") %>%
  set_mode("classification") -> forest_model2

forest_model2 %>% translate()

# Train the model
forest_model2 %>%
  fit(transmembrane ~ ., data=training(split_data)) -> forest_fit2 #predict transmembrane, given everything else

forest_fit2

# Test the model
forest_fit2 %>%
  predict(testing(split_data))

forest_fit2 %>%
  predict(testing(split_data)) %>%
  bind_cols(testing(split_data)) -> prediction_results2

# Evaluate the predictions
prediction_results2 %>%
  group_by(transmembrane, .pred_class) %>%
  count() #gives counts



res1_sens 
prediction_results %>%
  sens(transmembrane, .pred_class) #gives sensitivity
res1_spec 
prediction_results %>%
  spec(transmembrane, .pred_class) #gives specificity
res1_met 
prediction_results %>%
  metrics(transmembrane, .pred_class) #gives accuracy and Cohen's kappa value

res2_sens 
prediction_results2 %>%
  sens(transmembrane, .pred_class) #gives sensitivity
res2_spec
prediction_results2 %>%
  spec(transmembrane, .pred_class) #gives specificity
res2_met 
prediction_results2 %>%
  metrics(transmembrane, .pred_class) #gives accuracy and Cohen's kappa value




res1_sens <- prediction_results %>%
  sens(transmembrane, .pred_class) #gives sensitivity
res1_spec <- prediction_results %>%
  spec(transmembrane, .pred_class) #gives specificity
res1_met <- prediction_results %>%
  metrics(transmembrane, .pred_class) #gives accuracy and Cohen's kappa value

res2_sens <- prediction_results2 %>%
  sens(transmembrane, .pred_class) #gives sensitivity
res2_spec <- prediction_results2 %>%
  spec(transmembrane, .pred_class) #gives specificity
res2_met <- prediction_results2 %>%
  metrics(transmembrane, .pred_class) #gives accuracy and Cohen's kappa value


#### Exercise 4 ####
# Initial steps are the same as above, copied for reference.

#### Load required R packages ####
library(tidyverse)
library(tidymodels)
tidymodels_prefer()

#### Load input data ####
read_delim("../../MachineLearningData/transmembrane_data.txt") -> data

#### Preparing data for modelling ####
# Turning transmembrane into a factor
data %>%
  mutate(
    transmembrane = factor(transmembrane)
  ) -> data

# Remove gene_id column
data %>%
  select(-gene_id) -> data

# Shuffle the rows
data %>%
  sample_frac() -> data

# Remove missing values
data %>%
  na.omit() -> data

# Split the data
data %>%
  initial_split(prop=0.8, strata=transmembrane) -> split_data

### New from here ###
# Build a recipe
recipe(transmembrane ~ .,
       data = training(split_data)) -> neural_recipe

neural_recipe

# Add processing steps to our recipe: 
# 1: log transform gene_length and transcript_length
# 2: z-score normalise numeric columns
# 3: turn all text columns into dummy number columns
neural_recipe %>%
  step_log(gene_length, transcript_length) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) -> neural_recipe
  
neural_recipe

# Build the model
mlp(
  epochs = 1000, #rounds of refinement
  hidden_units = 10, #nodes in hidden layer
  penalty = 0.01, #penalises complexity to prevent overfitting
  learn_rate = 0.01 #how much estimates are moved to optimise model
) %>%
  set_engine("brulee", validation = 0) %>%
  set_mode("classification") -> nnet_model

nnet_model %>% translate()

# Build a workflow
workflow() %>%
  add_recipe(neural_recipe) %>%
  add_model(nnet_model) -> neural_workflow

neural_workflow

# Train model via the workflow

fit(neural_workflow, data=training(split_data)) -> neural_fit

neural_fit

# Evaluate the model
predict(neural_fit, new_data = testing(split_data)) %>%
  bind_cols(testing(split_data)) %>%
  select(.pred_class, transmembrane) -> neural_predictions

neural_predictions

# count up number of correct and incorrect predictions
neural_predictions %>%
  group_by(.pred_class, transmembrane) %>%
  count()

# can make into nicer looking table
neural_predictions %>%
  group_by(.pred_class, transmembrane) %>%
  count() %>%
  pivot_wider(names_from = .pred_class,
              values_from = n,
              names_prefix = "predicted_") %>%
  rename(true_transmembrane = transmembrane)

# calculate specific metrics
neural_predictions %>%
  metrics(transmembrane, .pred_class)

neural_predictions %>%
  sens(transmembrane, .pred_class)

neural_predictions %>%
  spec(transmembrane, .pred_class)


#### Additional exercise ####
# Set up knn model
nearest_neighbor(neighbors = tune(), weight_func = "triangular") %>%
  set_mode("classification") %>%
  set_engine("kknn") -> model

# set up 10 fold cross validation split
vfold_cv(data, v=10) -> vdata

# build recipe and workflow
recipe(transmembrane ~ .,
       data = training(split_data)) -> knn_recipe

knn_recipe %>%
  step_log(gene_length, transcript_length) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) -> knn_recipe

workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(model) -> knn_workflow

knn_workflow

# look at tunable parameters
knn_workflow %>%
  extract_parameter_set_dials()

knn_workflow %>%
  extract_parameter_set_dials() %>%
  update(neighbors = neighbors(c(1,50))) -> tune_parameters

# run the workflow
knn_workflow %>%
  tune_grid(vdata, 
            grid = grid_regular(tune_parameters, levels = 20),
            metrics = metric_set(sens, spec)) -> tune_results

autoplot(tune_results)
###############################################################################################

# Pick best neighbours result and add to model, then rerun workflow code and get predictions
nearest_neighbor(neighbors = 5, weight_func = "triangular") %>%
  set_mode("classification") %>%
  set_engine("kknn") -> model

workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(model) -> knn_workflow

# Train model via the workflow

knn_workflow %>%
  fit_resamples(vdata) -> knn_fit

knn_fit

collect_metrics(knn_fit)

# Evaluate the model
predict(knn_fit, new_data = testing(vdata)) %>%
  bind_cols(testing(vdata)) %>%
  select(.pred_class, transmembrane) -> knn_predictions

knn_predictions

# count up number of correct and incorrect predictions
knn_predictions %>%
  group_by(.pred_class, transmembrane) %>%
  count()

#OR do you then just do the usual train/test approach?
# Train model via the workflow

fit(knn_workflow, data=training(split_data)) -> knn_fit

knn_fit

# Evaluate the model
predict(knn_fit, new_data = testing(split_data)) %>%
  bind_cols(testing(split_data)) %>%
  select(.pred_class, transmembrane) -> knn_predictions

knn_predictions

# count up number of correct and incorrect predictions
knn_predictions %>%
  group_by(.pred_class, transmembrane) %>%
  count()

# can make into nicer looking table
knn_predictions %>%
  group_by(.pred_class, transmembrane) %>%
  count() %>%
  pivot_wider(names_from = .pred_class,
              values_from = n,
              names_prefix = "predicted_") %>%
  rename(true_transmembrane = transmembrane)

# calculate specific metrics
knn_predictions %>%
  metrics(transmembrane, .pred_class)

knn_predictions %>%
  sens(transmembrane, .pred_class)

knn_predictions %>%
  spec(transmembrane, .pred_class)

