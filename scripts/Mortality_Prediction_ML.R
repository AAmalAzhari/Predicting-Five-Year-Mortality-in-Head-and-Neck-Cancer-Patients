# ---- setup ----
library(tidyverse)
library(caret)
library(recipes)
library(pROC)
library(DescTools)
library(corrplot)
library(knitr)
library(here)

# Packages used by caret / recipes behind the scenes: fail early with a clear message
needed <- c("glmnet", "pls", "klaR", "pamr", "gbm", "xgboost", "nnet", "RSNNS", "gower")
not_installed <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
if (length(not_installed)) stop("Please install: ", paste(not_installed, collapse = ", "))

theme_set(theme_minimal(base_size = 12))
set.seed(1234)

f2 <- function(x) formatC(x, format = "f", digits = 2)
f3 <- function(x) formatC(x, format = "f", digits = 3)

# Optional parallel processing (leave FALSE if you hit problems on Windows)
use_parallel <- FALSE
if (use_parallel) {
  library(doParallel)
  cl <- makePSOCKcluster(max(1, parallel::detectCores() - 1))
  registerDoParallel(cl)
}

# ---- read-data ----
DF <- read.csv(here::here("data", "synthetic_hnscc_data.csv"))
dim(DF)
str(DF[, 1:12])

# ---- target ----
y_col <- "death5yr"
DF[[y_col]] <- factor(DF[[y_col]], levels = c(1, 0), labels = c("Died", "Survived"))
levels(DF[[y_col]])
table(DF[[y_col]])
round(prop.table(table(DF[[y_col]])), 3)

# ---- features ----
# Categorical (nominal) features
x_bin <- c("Sex", "HPV")
x_nom <- c("Country", "Smoking", "Tumor_subtype")
cat_cols <- c(x_bin, x_nom)

# Ordinal feature
ord_cols <- "ECOG"

# Numeric features: everything else except ID, categorical, ordinal and the target
num_cols <- setdiff(names(DF), c("ID", cat_cols, ord_cols, y_col))

X_cols <- c(cat_cols, ord_cols, num_cols)
c(categorical = length(cat_cols), ordinal = length(ord_cols), numeric = length(num_cols), total = length(X_cols))

# ---- missing ----
miss_tbl <- DF %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Missing") %>%
  mutate(Percent = round(100 * Missing / nrow(DF), 1)) %>%
  filter(Missing > 0) %>%
  arrange(desc(Missing))
kable(miss_tbl, caption = "Variables with missing values")

# ---- missing-plot ----
ggplot(miss_tbl, aes(reorder(Variable, Percent), Percent)) +
  geom_col(fill = "#2C7FB8") + coord_flip() +
  labs(x = NULL, y = "% missing")

# ---- limits ----
age_min <- 18;  age_max <- 110     # adult cohort; ages above 110 are impossible
bmi_min <- 10;  bmi_max <- 80      # BMI outside this range is not compatible with life
py_min  <- 0;   py_max  <- 200     # pack-years cannot be negative or implausibly large

limits <- tibble(Variable = c("Age", "BMI", "Pack_years"),
                 Lower = c(age_min, bmi_min, py_min),
                 Upper = c(age_max, bmi_max, py_max))

impossible <- pmap_dfr(limits, function(Variable, Lower, Upper) {
  x   <- DF[[Variable]]
  bad <- !is.na(x) & (x < Lower | x > Upper)
  tibble(Variable, Lower, Upper, n_impossible = sum(bad),
         values = paste(sort(unique(x[bad])), collapse = ", "))
})
kable(impossible, caption = "Impossible values detected, by clinical limits")

# ---- clean-helper ----
# Same rules as the recipe, used only for exploratory plots
clean_impossible <- function(d) {
  d %>% mutate(
    Age        = ifelse(Age < age_min | Age > age_max, NA, Age),
    BMI        = ifelse(BMI < bmi_min | BMI > bmi_max, NA, BMI),
    Pack_years = ifelse(Pack_years < py_min | Pack_years > py_max, NA, Pack_years)
  )
}

# ---- split ----
set.seed(1234)
idx   <- caret::createDataPartition(DF[[y_col]], p = 0.80, list = FALSE)
tr_DF <- DF[idx, , drop = FALSE]
te_DF <- DF[-idx, , drop = FALSE]

rbind(train = c(n = nrow(tr_DF), prop.table(table(tr_DF[[y_col]]))),
      test  = c(n = nrow(te_DF), prop.table(table(te_DF[[y_col]])))) %>%
  round(3) %>% kable(caption = "Size and class balance of each set")

# ---- eda-desc-core ----
DescTools::Desc(tr_DF[, c("Age", "BMI", "Pack_years")], plotit = TRUE)

# ---- eda-desc-more ----
DescTools::Desc(tr_DF[, c("CRP", "Tumor_volume", "Albumin", "Smoking", "ECOG", "Tumor_subtype")], plotit = TRUE)

# ---- eda-desc-target ----
DescTools::Desc(death5yr ~ Age + Smoking + Tumor_subtype, data = clean_impossible(tr_DF), plotit = TRUE)

# ---- eda-class-plots ----
clean_tr <- clean_impossible(tr_DF)
clean_tr %>%
  select(all_of(y_col), Age, Tumor_volume, Cellcycle_score, Albumin, Shape_ScaledSphericity) %>%
  pivot_longer(-all_of(y_col), names_to = "Feature", values_to = "Value") %>%
  ggplot(aes(Value, fill = .data[[y_col]])) +
  geom_density(alpha = .5) +
  facet_wrap(~Feature, scales = "free") +
  labs(fill = "5-year outcome", y = "Density")

# ---- eda-cat-plots ----
tr_DF %>%
  select(all_of(c(y_col, "Smoking", "Tumor_subtype", "HPV", "ECOG"))) %>%
  mutate(across(c(HPV, ECOG), as.character)) %>%
  pivot_longer(-all_of(y_col), names_to = "Feature", values_to = "Level") %>%
  ggplot(aes(Level, fill = .data[[y_col]])) +
  geom_bar(position = "fill") +
  facet_wrap(~Feature, scales = "free_x") +
  labs(y = "Proportion", fill = "5-year outcome")

# ---- eda-skew ----
skew_tbl <- clean_tr %>%
  summarise(across(all_of(num_cols), ~ DescTools::Skew(.x, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "Variable", values_to = "Skewness") %>%
  arrange(desc(abs(Skewness)))
kable(head(skew_tbl, 8), digits = 2, caption = "Most skewed numeric predictors (candidates for Yeo-Johnson)")

# ---- eda-corr ----
M <- cor(clean_tr[, num_cols], use = "pairwise.complete.obs")
corrplot(M, method = "color", type = "upper", order = "hclust",
         tl.cex = 0.55, tl.col = "black", cl.cex = 0.7)

as.data.frame(as.table(M)) %>%
  filter(as.integer(Var1) < as.integer(Var2), abs(Freq) > 0.7) %>%
  arrange(desc(abs(Freq))) %>%
  kable(digits = 2, col.names = c("Variable 1", "Variable 2", "Correlation"),
        caption = "Highly correlated pairs (|r| > 0.7): the reason for a correlation filter")

# ---- mod-form ----
mod_form <- as.formula(paste0(y_col, " ~ ."))
mod_form

# ---- recipe ----
rec <- recipe(mod_form, data = tr_DF) |>
  step_rm(ID) |>

  # Types, and impossible values -> NA so that they are imputed
  step_mutate(
    HPV  = as.character(HPV),
    ECOG = ordered(as.character(ECOG), levels = c("0", "1", "2", "3")),
    Age        = ifelse(Age < !!age_min | Age > !!age_max, NA, Age),
    BMI        = ifelse(BMI < !!bmi_min | BMI > !!bmi_max, NA, BMI),
    Pack_years = ifelse(Pack_years < !!py_min | Pack_years > !!py_max, NA, Pack_years)
  ) |>

  # Categorical safeguards
  step_string2factor(all_of(cat_cols)) |>
  step_novel(all_of(cat_cols), new_level = "NewLvl") |>
  step_unknown(all_of(cat_cols), new_level = "Unknown") |>
  step_other(all_of(cat_cols), threshold = 0.01, other = "OtherLvl") |>

  # Multivariate KNN imputation
  step_impute_knn(all_predictors(), neighbors = 5) |>

  # Ordinal score, then dummy encoding
  step_ordinalscore(all_of(ord_cols)) |>
  step_dummy(all_nominal_predictors(), one_hot = FALSE) |>

  # Stability filters
  step_zv(all_predictors()) |>
  step_nzv(all_predictors()) |>

  # Skewness correction for continuous / ordinal features only (any_of: tolerant if a step removed a column)
  step_YeoJohnson(any_of(num_cols), any_of(ord_cols)) |>

  # Centre and scale everything
  step_center(all_numeric_predictors()) |>
  step_scale(all_numeric_predictors()) |>

  # Multicollinearity filter
  step_corr(all_numeric_predictors(), threshold = 0.95)

rec

# ---- recipe-check ----
rec_prep <- prep(rec, training = tr_DF)
baked    <- bake(rec_prep, new_data = NULL)

step_ids <- function(cls) which(map_chr(rec_prep$steps, ~ class(.x)[1]) == cls)
dropped  <- function(cls) tidy(rec_prep, number = step_ids(cls))$terms

tibble(Item = c("Predictors before pre-processing", "Predictors after pre-processing",
                "Missing values remaining", "Removed by near-zero variance filter",
                "Removed by correlation filter"),
       Value = c(length(X_cols), ncol(baked) - 1, sum(is.na(baked)),
                 paste(dropped("step_nzv"), collapse = ", ") %>% ifelse(. == "", "none", .),
                 paste(dropped("step_corr"), collapse = ", ") %>% ifelse(. == "", "none", .))) %>%
  kable(caption = "Effect of the pre-processing recipe on the training set")

# ---- x-validation ----
set.seed(1234)
folds <- caret::createFolds(tr_DF[[y_col]], k = 5, returnTrain = TRUE)

# AUC, sensitivity, specificity (twoClassSummary) plus accuracy and kappa
fiveStats <- function(data, lev = NULL, model = NULL) {
  c(caret::twoClassSummary(data, lev, model), caret::defaultSummary(data, lev, model))
}

ctrl <- caret::trainControl(
  method = "cv", number = 5, index = folds,
  classProbs = TRUE, summaryFunction = fiveStats,
  search = "grid", selectionFunction = "best",
  savePredictions = "final", verboseIter = FALSE,
  allowParallel = use_parallel
)

# ---- model-specs ----
model_specs <- list(
  glmnet = list(method = "glmnet",
                grid = expand.grid(alpha = c(0, 0.25, 0.5, 0.75, 1),
                                   lambda = 10^seq(-4, -0.5, length.out = 10)),
                args = list()),
  pls    = list(method = "pls",
                grid = data.frame(ncomp = 1:12),
                args = list()),
  rda    = list(method = "rda",
                grid = expand.grid(gamma = c(0, 0.25, 0.5, 0.75, 1),
                                   lambda = c(0.25, 0.5, 0.75, 1)),
                args = list()),
  nb     = list(method = "nb",
                grid = expand.grid(fL = c(0, 1), usekernel = c(TRUE, FALSE), adjust = c(0.5, 1, 1.5)),
                args = list()),
  knn    = list(method = "knn",
                grid = data.frame(k = seq(5, 45, by = 4)),
                args = list()),
  pam    = list(method = "pam",
                grid = data.frame(threshold = seq(0, 3, length.out = 15)),
                args = list()),
  gbm    = list(method = "gbm",
                grid = expand.grid(n.trees = c(100, 200, 300), interaction.depth = 1:3,
                                   shrinkage = c(0.01, 0.05, 0.1), n.minobsinnode = c(10, 20)),
                args = list(verbose = FALSE)),
  xgbTree = list(method = "xgbTree",
                 grid = expand.grid(nrounds = c(100, 200), max_depth = c(2, 3, 4),
                                    eta = c(0.03, 0.1), gamma = 0,
                                    colsample_bytree = c(0.6, 0.8), min_child_weight = c(1, 5),
                                    subsample = c(0.7, 0.9)),
                 args = list()),
  nnet   = list(method = "nnet",
                grid = expand.grid(size = c(1, 3, 5, 7), decay = c(0.01, 0.1, 1)),
                args = list(trace = FALSE, maxit = 300, MaxNWts = 5000)),
  mlp    = list(method = "mlp",
                grid = data.frame(size = c(1, 3, 5, 7)),
                args = list())
)

fit_one <- function(spec) {
  set.seed(1234)                              # same seed for every model
  do.call(caret::train,
          c(list(x = rec, data = tr_DF, method = spec$method, metric = "ROC",
                 trControl = ctrl, tuneGrid = spec$grid),
            spec$args))
}

# ---- train ----
dir.create(here::here("output"), showWarnings = FALSE)
models_path <- here::here("output", "models.rds")

if (file.exists(models_path)) {
  models <- readRDS(models_path)
} else {
  models <- lapply(model_specs, fit_one)
  saveRDS(models, models_path)
}
names(models)

# ---- cv-compare ----
cv_tbl <- map_dfr(models, caret::getTrainPerf, .id = "Model") %>%
  transmute(Model, ROC = TrainROC, Sensitivity = TrainSens, Specificity = TrainSpec,
            Accuracy = TrainAccuracy, Kappa = TrainKappa) %>%
  arrange(desc(ROC))
kable(cv_tbl, digits = 3, caption = "5-fold cross-validated performance of the tuned models (training set)")

# ---- cv-plot ----
rs <- caret::resamples(models)
rs$values %>%
  select(Resample, ends_with("~ROC")) %>%
  pivot_longer(-Resample, names_to = "Model", values_to = "ROC") %>%
  mutate(Model = sub("~ROC", "", Model)) %>%
  ggplot(aes(reorder(Model, ROC, median), ROC)) +
  geom_boxplot(fill = "#9ECAE1") +
  geom_jitter(width = .1, alpha = .6) +
  coord_flip() +
  labs(x = NULL, y = "Cross-validated AUC")

# ---- best-tune ----
best_tune <- map_dfr(models, function(m) {
  bt <- m$bestTune
  tibble(`Selected hyper-parameters` = paste(names(bt), unlist(bt), sep = " = ", collapse = ", "))
}, .id = "Model")
kable(best_tune, caption = "Hyper-parameters selected by cross-validation")

# ---- perf-helper ----
perf_metrics <- function(fit, newdata) {
  prob <- predict(fit, newdata, type = "prob")[["Died"]]
  cls  <- predict(fit, newdata)
  cm   <- caret::confusionMatrix(cls, newdata[[y_col]], positive = "Died")
  auc  <- as.numeric(pROC::auc(pROC::roc(newdata[[y_col]], prob,
                                         levels = c("Survived", "Died"), direction = "<", quiet = TRUE)))
  tibble(Accuracy = cm$overall[["Accuracy"]],
         `Balanced accuracy` = cm$byClass[["Balanced Accuracy"]],
         Sensitivity = cm$byClass[["Sensitivity"]],
         Specificity = cm$byClass[["Specificity"]],
         PPV = cm$byClass[["Pos Pred Value"]],
         NPV = cm$byClass[["Neg Pred Value"]],
         Kappa = cm$overall[["Kappa"]],
         AUC = auc)
}

# ---- perf-tables ----
perf_te <- map_dfr(models, ~ perf_metrics(.x, te_DF), .id = "Model") %>% arrange(desc(AUC))
perf_tr <- map_dfr(models, ~ perf_metrics(.x, tr_DF), .id = "Model") %>% arrange(desc(AUC))

kable(perf_te, digits = 3, caption = "Test-set performance (unseen data; threshold 0.5)")
kable(perf_tr, digits = 3, caption = "Training-set performance (apparent performance)")

# ---- overfit-plot ----
bind_rows(Train = perf_tr, Test = perf_te, .id = "Set") %>%
  ggplot(aes(reorder(Model, AUC), AUC, fill = Set)) +
  geom_col(position = "dodge") +
  coord_flip(ylim = c(0.5, 1)) +
  labs(x = NULL, y = "AUC")

# ---- final-select ----
best_name <- cv_tbl$Model[which.max(cv_tbl$ROC)]
fit_final <- models[[best_name]]
best_name
fit_final

# ---- final-predict ----
tr_class <- predict(fit_final, newdata = tr_DF)
tr_prob  <- predict(fit_final, newdata = tr_DF, type = "prob")[["Died"]]
te_class <- predict(fit_final, newdata = te_DF)
te_prob  <- predict(fit_final, newdata = te_DF, type = "prob")[["Died"]]

roc_tr <- pROC::roc(tr_DF[[y_col]], tr_prob, levels = c("Survived", "Died"), direction = "<", quiet = TRUE)
roc_te <- pROC::roc(te_DF[[y_col]], te_prob, levels = c("Survived", "Died"), direction = "<", quiet = TRUE)
ci_te  <- pROC::ci.auc(roc_te)

cm_tr <- caret::confusionMatrix(tr_class, tr_DF[[y_col]], positive = "Died")
cm_te <- caret::confusionMatrix(te_class, te_DF[[y_col]], positive = "Died")

bind_rows(Train = perf_metrics(fit_final, tr_DF), Test = perf_metrics(fit_final, te_DF), .id = "Set") %>%
  kable(digits = 3, caption = "Final model: performance at the default 0.5 threshold")

# ---- final-roc ----
pROC::ggroc(list(Train = roc_tr, Test = roc_te), linewidth = 1) +
  geom_abline(intercept = 1, slope = 1, linetype = "dashed", colour = "grey50") +
  coord_equal() +
  labs(x = "Specificity", y = "Sensitivity", colour = NULL)

# ---- final-cm ----
cat("--- Test set confusion matrix ---\n")
print(cm_te)

# ---- final-calibration ----
calib <- bind_rows(
  tibble(Set = "Train", prob = tr_prob, obs = as.integer(tr_DF[[y_col]] == "Died")),
  tibble(Set = "Test",  prob = te_prob, obs = as.integer(te_DF[[y_col]] == "Died"))
) %>%
  group_by(Set) %>%
  mutate(bin = ntile(prob, 10)) %>%
  group_by(Set, bin) %>%
  summarise(predicted = mean(prob), observed = mean(obs), .groups = "drop")

ggplot(calib, aes(predicted, observed, colour = Set)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", colour = "grey50") +
  geom_line() + geom_point(size = 2) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Mean predicted probability of death", y = "Observed proportion who died")

# ---- final-importance ----
vi <- tryCatch(caret::varImp(fit_final), error = function(e) NULL)
if (!is.null(vi)) {
  plot(vi, top = 15)
} else {
  message("Variable importance is not available for this model type.")
}

# ---- final-threshold ----
thr <- pROC::coords(roc_tr, x = "best", best.method = "youden",
                    ret = "threshold", transpose = FALSE)
thr <- as.numeric(thr$threshold)[1]

te_class_thr <- factor(ifelse(te_prob >= thr, "Died", "Survived"), levels = c("Died", "Survived"))
cm_te_thr <- caret::confusionMatrix(te_class_thr, te_DF[[y_col]], positive = "Died")

tibble(Threshold = c(0.5, thr),
       Sensitivity = c(cm_te$byClass[["Sensitivity"]], cm_te_thr$byClass[["Sensitivity"]]),
       Specificity = c(cm_te$byClass[["Specificity"]], cm_te_thr$byClass[["Specificity"]]),
       Accuracy    = c(cm_te$overall[["Accuracy"]], cm_te_thr$overall[["Accuracy"]])) %>%
  kable(digits = 3, caption = "Test-set performance at the default and the Youden threshold")

# ---- session ----
if (use_parallel) parallel::stopCluster(cl)
sessionInfo()
