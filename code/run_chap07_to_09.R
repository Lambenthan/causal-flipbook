#!/usr/bin/env Rscript
# 在 RHC 数据（29 协变量）上跑第 7–9 章：DML / TMLE / E-value / sensemakr / 因果森林
# 设计目标：单次执行 < 5 分钟。SuperLearner 仅用 glm + glmnet + mean，避免 ranger 拖慢。

suppressPackageStartupMessages({
  library(tidyverse)
  library(SuperLearner)
  library(DoubleML)
  library(mlr3)
  library(mlr3learners)
  library(tmle)
  library(EValue)
  library(sensemakr)
  library(grf)
  library(here)
})

set.seed(2026)

d <- read_csv(here("data", "rhc.csv"), show_col_types = FALSE) |>
  mutate(death180_bin = if_else(death180 == "Yes", 1L, 0L),
         rhc_bin      = if_else(rhc == 1, 1L, 0L),
         sex_bin      = if_else(sex == "Male", 1L, 0L),
         dnr_bin      = if_else(dnr_status == "Yes", 1L, 0L),
         cancer       = as.integer(as.factor(cancer)) - 1L)  # 0=No, 1=其他

covs <- c("age", "sex_bin", "edu", "das_index", "apache_score",
          "glasgow_coma_score", "blood_pressure", "wbc", "heart_rate",
          "respiratory_rate", "temperature", "albumin", "hematocrit",
          "bilirubin", "creatinine", "weight",
          "cancer", "cardiovascular", "congestive_hf", "dementia",
          "psychiatric", "pulmonary", "renal", "hepatic",
          "gi_bleed", "tumor", "immunosupperssion", "transfer_hx", "mi")

X <- as.matrix(d[, covs])
A <- d$rhc_bin
Y <- d$death180_bin
n <- nrow(d)

cat("Sample size:", n, "  covariates:", length(covs), "\n\n")

# ---------- 第 7 章 · DML ----------
cat("[DML] running DoubleML with ranger learners (5-fold CV) ...\n")
df_dml <- DoubleMLData$new(
  data = data.frame(Y = Y, A = A, X),
  y_col = "Y", d_cols = "A", x_cols = covs)

lgr::get_logger("mlr3")$set_threshold("warn")

lrn_g <- lrn("regr.ranger", num.trees = 300, min.node.size = 5)
lrn_m <- lrn("classif.ranger", num.trees = 300, min.node.size = 5,
             predict_type = "prob")

dml_irm <- DoubleMLIRM$new(
  data = df_dml,
  ml_g = lrn_g,
  ml_m = lrn_m,
  n_folds = 5,
  score = "ATE")
dml_irm$fit()

dml_rd <- dml_irm$coef[1]
dml_se <- dml_irm$se[1]
dml_ci <- c(dml_rd - 1.96 * dml_se, dml_rd + 1.96 * dml_se)
cat(sprintf("[DML]  RD = %.4f  SE = %.4f  CI = [%.4f, %.4f]\n",
            dml_rd, dml_se, dml_ci[1], dml_ci[2]))

# ---------- 第 7 章 · TMLE（手算 targeting step；用 ranger 拟合 Q 与 g 以演示 ML 嵌入） ----------
cat("\n[TMLE] computing TMLE manually (ranger Q + ranger g + targeting step) ...\n")

library(ranger)
df_full <- data.frame(Y = Y, A = A, X)

# Step 1+2: 5 折交叉拟合 Q(A,L) 与 g(L)，避免 in-sample 过拟合让 IC 方差被压缩
trim <- function(p) pmin(pmax(p, 0.005), 0.995)

set.seed(2026)
K <- 5
folds <- sample(rep(1:K, length.out = n))
Q0_AL <- numeric(n); Q0_1L <- numeric(n); Q0_0L <- numeric(n); ps <- numeric(n)

for (k in 1:K) {
  tr <- which(folds != k); te <- which(folds == k)
  rf_Q <- ranger(Y ~ ., data = df_full[tr, ], num.trees = 500, min.node.size = 5,
                 classification = FALSE, seed = 2026 + k)
  Q0_AL[te] <- predict(rf_Q, df_full[te, ])$predictions
  df_1_te <- df_full[te, ]; df_1_te$A <- 1
  df_0_te <- df_full[te, ]; df_0_te$A <- 0
  Q0_1L[te] <- predict(rf_Q, df_1_te)$predictions
  Q0_0L[te] <- predict(rf_Q, df_0_te)$predictions

  rf_g <- ranger(factor(A) ~ ., data = df_full[tr, c("A", covs)],
                 num.trees = 500, min.node.size = 5,
                 probability = TRUE, seed = 2026 + k)
  ps[te] <- predict(rf_g, df_full[te, c("A", covs)])$predictions[, "1"]
}

Q0_AL <- trim(Q0_AL); Q0_1L <- trim(Q0_1L); Q0_0L <- trim(Q0_0L); ps <- trim(ps)
cat(sprintf("Cross-fit Q: mean Q(1) = %.4f, mean Q(0) = %.4f, naive RD = %.4f\n",
            mean(Q0_1L), mean(Q0_0L), mean(Q0_1L) - mean(Q0_0L)))
cat(sprintf("Cross-fit PS: range [%.4f, %.4f], mean treated = %.4f, mean control = %.4f\n",
            min(ps), max(ps), mean(ps[A == 1]), mean(ps[A == 0])))

# Step 3: clever covariate H(A, L) = A/g(L) - (1-A)/(1-g(L))
H_AL <- A / ps - (1 - A) / (1 - ps)
H_1L <- 1 / ps
H_0L <- -1 / (1 - ps)

# Step 4: 在 logit(Q0) 上拟合 epsilon
logit <- function(p) log(p / (1 - p))
expit <- function(x) 1 / (1 + exp(-x))
eps_fit <- glm(Y ~ -1 + offset(logit(Q0_AL)) + H_AL, family = binomial)
eps <- coef(eps_fit)[1]
cat(sprintf("Targeting epsilon = %.6f\n", eps))

# Step 5: 更新 Q1
Q1_1L <- expit(logit(Q0_1L) + eps * H_1L)
Q1_0L <- expit(logit(Q0_0L) + eps * H_0L)

tmle_rd <- mean(Q1_1L) - mean(Q1_0L)

# Step 6: Influence function -based SE
Q1_AL <- ifelse(A == 1, Q1_1L, Q1_0L)
IC <- (A / ps - (1 - A) / (1 - ps)) * (Y - Q1_AL) +
      (Q1_1L - Q1_0L) - tmle_rd
tmle_se <- sd(IC) / sqrt(n)
tmle_ci <- c(tmle_rd - 1.96 * tmle_se, tmle_rd + 1.96 * tmle_se)
tmle_var <- tmle_se^2
cat(sprintf("[TMLE] RD = %.4f  SE = %.4f  CI = [%.4f, %.4f]\n",
            tmle_rd, tmle_se, tmle_ci[1], tmle_ci[2]))

# 报告 Super Learner 在 Q 上的折-内权重（n=2000 子样本演示集成）
cat("\n[SL] SuperLearner CV weights on n=2000 subsample (glm + glmnet + ranger + mean) ...\n")
sub_idx <- sample(n, 2000)
sl_lib_full <- c("SL.glm", "SL.glmnet", "SL.ranger", "SL.mean")
sl_q <- SuperLearner(Y = Y[sub_idx],
                     X = data.frame(A = A[sub_idx], X[sub_idx, ]),
                     family = binomial(),
                     SL.library = sl_lib_full,
                     cvControl = list(V = 5))
print(sl_q$coef)
sl_weights <- sl_q$coef

# ---------- 第 8 章 · E-value ----------
cat("\n[E-value] computing E-value for adjusted estimates ...\n")
cat("Using chap03 regression OR = 1.2876, CI = [1.1340, 1.4620]\n")
ev_or <- evalues.OR(est = 1.2876, lo = 1.1340, hi = 1.4620, rare = FALSE)
print(ev_or)

cat("\nUsing chap06 AIPW RD = 0.0455 → adjusted RR\n")
p1 <- mean(Y[A == 1]); p0 <- mean(Y[A == 0])
adj_rr    <- (p0 + 0.0455) / p0
adj_rr_lo <- (p0 + 0.0181) / p0
adj_rr_hi <- (p0 + 0.0729) / p0
cat(sprintf("p1 = %.4f, p0 = %.4f, adjusted RR = %.4f, CI = [%.4f, %.4f]\n",
            p1, p0, adj_rr, adj_rr_lo, adj_rr_hi))
ev_rr <- evalues.RR(est = adj_rr, lo = adj_rr_lo, hi = adj_rr_hi)
print(ev_rr)

# ---------- 第 8 章 · sensemakr ----------
cat("\n[sensemakr] OLS sensitivity, benchmark = apache_score, kd = 1,2,3 ...\n")
fml_y_full <- as.formula(paste("death180_bin ~ rhc_bin +", paste(covs, collapse = " + ")))
m_lm <- lm(fml_y_full, data = d)
sens <- sensemakr(model = m_lm, treatment = "rhc_bin",
                  benchmark_covariates = "apache_score",
                  kd = c(1, 2, 3))
print(summary(sens))

# 关键诊断
cat(sprintf("\nRobustness value (q=1, alpha=0.05): %.4f\n", sens$sensitivity_stats$rv_qa))
cat(sprintf("Partial R^2 (treatment with outcome): %.4f\n",
            sens$sensitivity_stats$r2yd.x))

# ---------- 第 9 章 · 因果森林 ----------
cat("\n[CF] training causal forest with 2000 trees (29 covariates) ...\n")
cf <- causal_forest(X = X, Y = Y, W = A, num.trees = 2000, seed = 2026)

cf_ate <- average_treatment_effect(cf, target.sample = "all")
cat(sprintf("[CF ATE] RD = %.4f  SE = %.4f\n", cf_ate[1], cf_ate[2]))

tau_hat <- predict(cf)$predictions
cat(sprintf("[CATE] mean = %.4f  sd = %.4f\n", mean(tau_hat), sd(tau_hat)))
cat(sprintf("[CATE] min = %.4f  q10 = %.4f  q25 = %.4f  median = %.4f  q75 = %.4f  q90 = %.4f  max = %.4f\n",
            min(tau_hat), quantile(tau_hat, 0.10), quantile(tau_hat, 0.25),
            median(tau_hat),
            quantile(tau_hat, 0.75), quantile(tau_hat, 0.90), max(tau_hat)))
cat(sprintf("Pr(CATE > 0) = %.4f  Pr(CATE < 0) = %.4f  Pr(|CATE| < 0.02) = %.4f\n",
            mean(tau_hat > 0), mean(tau_hat < 0), mean(abs(tau_hat) < 0.02)))

vi <- variable_importance(cf)
vi_df <- tibble(var = covs, importance = as.numeric(vi)) |>
  arrange(desc(importance))
cat("\nTop 10 variable importance (causal forest):\n")
print(vi_df |> head(10))

d$tau_hat <- tau_hat
apache_q <- quantile(d$apache_score, c(0.25, 0.5, 0.75))
d$apache_grp <- cut(d$apache_score, breaks = c(-Inf, apache_q, Inf),
                    labels = c("Q1 (轻症)", "Q2", "Q3", "Q4 (重症)"))
cate_by_apache <- d |> group_by(apache_grp) |>
  summarise(n = n(), cate_mean = mean(tau_hat), cate_sd = sd(tau_hat))
cat("\nCATE by APACHE quartile:\n"); print(cate_by_apache)

age_q <- quantile(d$age, c(0.25, 0.5, 0.75))
d$age_grp <- cut(d$age, breaks = c(-Inf, age_q, Inf),
                 labels = c("Q1 (年轻)", "Q2", "Q3", "Q4 (老年)"))
cate_by_age <- d |> group_by(age_grp) |>
  summarise(n = n(), cate_mean = mean(tau_hat), cate_sd = sd(tau_hat))
cat("\nCATE by age quartile:\n"); print(cate_by_age)

blp <- best_linear_projection(cf, X[, c("age", "apache_score", "creatinine",
                                        "blood_pressure", "albumin")])
cat("\nBest Linear Projection (主要协变量):\n"); print(blp)

saveRDS(list(
  dml_rd = dml_rd, dml_se = dml_se, dml_ci = dml_ci,
  tmle_rd = tmle_rd, tmle_var = tmle_var, tmle_ci = tmle_ci,
  sl_weights = sl_weights,
  ev_or = ev_or, ev_rr = ev_rr,
  sens_summary = summary(sens),
  rv_qa = sens$sensitivity_stats$rv_qa,
  r2yd = sens$sensitivity_stats$r2yd.x,
  cf_ate = cf_ate,
  tau_hat = tau_hat,
  vi_df = vi_df,
  cate_by_apache = cate_by_apache,
  cate_by_age = cate_by_age,
  blp = blp
), "/tmp/rhc_estimates_chap789.rds")

cat("\n================ DONE ================\n")
