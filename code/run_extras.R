#!/usr/bin/env Rscript
# 为 patient-stories.md 和 anti-patterns.md 跑真实数字
# 输出：
#   /tmp/rhc_patient_stories.rds   5 位代表性患者的全方法估计
#   /tmp/rhc_anti_patterns.rds     5 类错误调整集下的偏差表

suppressPackageStartupMessages({
  library(tidyverse); library(MatchIt); library(WeightIt); library(grf)
})

set.seed(2026)
DATA_PATH <- "/Users/han/Desktop/AI_Plan/2026LLM/Media_Paper/CausalInferenceBook-v2/data/rhc.csv"

d <- read_csv(DATA_PATH, show_col_types = FALSE) |>
  mutate(death180_bin = if_else(death180 == "Yes", 1L, 0L),
         rhc_bin      = if_else(rhc == 1, 1L, 0L),
         sex_bin      = if_else(sex == "Male", 1L, 0L),
         dnr_bin      = if_else(dnr_status == "Yes", 1L, 0L),
         cancer       = as.integer(as.factor(cancer)) - 1L) |>
  mutate(row_id = row_number())

covs_correct <- c("age", "sex_bin", "edu", "das_index", "apache_score",
                  "glasgow_coma_score", "blood_pressure", "wbc", "heart_rate",
                  "respiratory_rate", "temperature", "albumin", "hematocrit",
                  "bilirubin", "creatinine", "weight",
                  "cancer", "cardiovascular", "congestive_hf", "dementia",
                  "psychiatric", "pulmonary", "renal", "hepatic",
                  "gi_bleed", "tumor", "immunosupperssion", "transfer_hx", "mi")

# ============================================================
# 第 1 部分：5 位代表性患者
# ============================================================
cat("[Patient Stories]\n")

# 为正确调整集训练好基础模型
fml_y <- as.formula(paste("death180_bin ~ rhc_bin +", paste(covs_correct, collapse = " + ")))
fml_a <- as.formula(paste("rhc_bin ~", paste(covs_correct, collapse = " + ")))
m_y <- glm(fml_y, data = d, family = binomial)
m_a <- glm(fml_a, data = d, family = binomial)
d$ps <- predict(m_a, type = "response")

# 反事实预测
d1 <- d |> mutate(rhc_bin = 1L)
d0 <- d |> mutate(rhc_bin = 0L)
d$mhat_1 <- predict(m_y, newdata = d1, type = "response")
d$mhat_0 <- predict(m_y, newdata = d0, type = "response")
d$g_cate <- d$mhat_1 - d$mhat_0

# AIPW 个体得分
d$aipw_score <- (d$mhat_1 - d$mhat_0) +
                (d$rhc_bin / d$ps) * (d$death180_bin - d$mhat_1) -
                ((1 - d$rhc_bin) / (1 - d$ps)) * (d$death180_bin - d$mhat_0)

# 因果森林 CATE
X <- as.matrix(d[, covs_correct])
cf <- causal_forest(X = X, Y = d$death180_bin, W = d$rhc_bin,
                    num.trees = 2000, seed = 2026)
d$cf_cate <- predict(cf)$predictions

# 选 5 位代表性患者
# 1. 老年重症 + 实际 RHC + 死亡
# 2. 中年中症 + 实际 RHC + 存活
# 3. 年轻重症 + 实际未 RHC + 存活
# 4. 老年轻症 + 实际未 RHC + 死亡
# 5. 中重症 + 实际 RHC + 死亡（与 1 形成对照）

pick <- function(filter_expr, sort_expr = NULL) {
  out <- d |> filter(!!rlang::parse_expr(filter_expr))
  if (!is.null(sort_expr)) out <- out |> arrange(!!rlang::parse_expr(sort_expr))
  out |> slice_head(n = 1)
}

p1 <- pick("age >= 75 & apache_score >= 70 & creatinine >= 2.0 & rhc_bin == 1 & death180_bin == 1",
           "desc(apache_score)")
p2 <- pick("age >= 45 & age <= 60 & apache_score >= 40 & apache_score <= 55 & rhc_bin == 1 & death180_bin == 0",
           "apache_score")
p3 <- pick("age <= 40 & apache_score >= 60 & rhc_bin == 0 & death180_bin == 0",
           "desc(apache_score)")
p4 <- pick("age >= 75 & apache_score <= 30 & rhc_bin == 0 & death180_bin == 1",
           "apache_score")
p5 <- pick("age >= 60 & age <= 75 & apache_score >= 50 & apache_score <= 65 & rhc_bin == 1 & death180_bin == 1 & blood_pressure <= 70",
           "blood_pressure")

patients <- bind_rows(
  p1 |> mutate(role = "1. 老年重症 + RHC + 死亡"),
  p2 |> mutate(role = "2. 中年中症 + RHC + 存活"),
  p3 |> mutate(role = "3. 年轻重症 + 未 RHC + 存活"),
  p4 |> mutate(role = "4. 老年轻症 + 未 RHC + 死亡"),
  p5 |> mutate(role = "5. 中老年血流不稳 + RHC + 死亡")
)

cat("\n选出的 5 位患者：\n")
patients |>
  select(role, row_id, age, apache_score, glasgow_coma_score,
         blood_pressure, creatinine, congestive_hf, cancer,
         rhc_bin, death180_bin, ps, mhat_0, mhat_1, g_cate, aipw_score, cf_cate) |>
  print(n = Inf, width = Inf)

saveRDS(list(
  patients = patients,
  ate_g = mean(d$g_cate),
  ate_aipw = mean(d$aipw_score),
  ate_cf = average_treatment_effect(cf, target.sample = "all")
), "/tmp/rhc_patient_stories.rds")

# ============================================================
# 第 2 部分：错误调整集画廊
# ============================================================
cat("\n[Anti-Patterns]\n")

# 通用：用给定 covs 跑 G / IPW / AIPW，返回 RD 与 SE
run_methods <- function(d, covs, label) {
  fml_y <- as.formula(paste("death180_bin ~ rhc_bin +", paste(covs, collapse = " + ")))
  fml_a <- as.formula(paste("rhc_bin ~", paste(covs, collapse = " + ")))

  m_y <- glm(fml_y, data = d, family = binomial)
  m_a <- glm(fml_a, data = d, family = binomial)
  ps <- predict(m_a, type = "response")
  ps <- pmin(pmax(ps, 0.005), 0.995)

  d1 <- d; d1$rhc_bin <- 1L
  d0 <- d; d0$rhc_bin <- 0L
  m1 <- predict(m_y, newdata = d1, type = "response")
  m0 <- predict(m_y, newdata = d0, type = "response")

  rd_g <- mean(m1 - m0)

  # IPW
  w <- ifelse(d$rhc_bin == 1, 1/ps, 1/(1-ps))
  rd_ipw <- weighted.mean(d$death180_bin[d$rhc_bin == 1], w[d$rhc_bin == 1]) -
            weighted.mean(d$death180_bin[d$rhc_bin == 0], w[d$rhc_bin == 0])

  # AIPW
  ic <- (m1 - m0) +
        (d$rhc_bin / ps) * (d$death180_bin - m1) -
        ((1 - d$rhc_bin) / (1 - ps)) * (d$death180_bin - m0)
  rd_aipw <- mean(ic)
  se_aipw <- sd(ic) / sqrt(nrow(d))

  tibble(label = label,
         rd_g = rd_g, rd_ipw = rd_ipw,
         rd_aipw = rd_aipw, se_aipw = se_aipw,
         max_w = max(w))
}

# 基线（正确调整集）
res_base <- run_methods(d, covs_correct, "正确：29 协变量")

# Anti-pattern A：完全不调整（粗差异路线）
res_A <- run_methods(d, c("sex_bin"), "A. 几乎不调整（仅 sex）")

# Anti-pattern B：漏掉 APACHE 与 GCS（最强混杂）
covs_no_severity <- setdiff(covs_correct, c("apache_score", "glasgow_coma_score"))
res_B <- run_methods(d, covs_no_severity, "B. 漏掉 APACHE 与 GCS")

# Anti-pattern C：仅控制人口学（age + sex + edu + race-like），不放临床
covs_demo <- c("age", "sex_bin", "edu")
res_C <- run_methods(d, covs_demo, "C. 仅控制人口学")

# Anti-pattern D：控制处理后变量（合成一个明显处理后的指标）
# 用 dnr_bin 作为代理 + 合成 z_post = 0.5*A + 0.4*Y_proxy + noise
set.seed(2026)
d_post <- d
y_proxy <- ifelse(d$apache_score > median(d$apache_score), 1, 0)
d_post$z_post <- 0.6 * d$rhc_bin + 0.4 * y_proxy + 0.2 * d$dnr_bin + rnorm(nrow(d), 0, 0.3)
covs_with_post <- c(covs_correct, "z_post")
res_D <- run_methods(d_post, covs_with_post, "D. 控制处理后变量 z_post")

# Anti-pattern E：控制对撞节点（合成一个 collider）
# z_collider = 0.5 * A + 0.5 * Y + noise，A 与 Y 共同的下游
d_coll <- d
d_coll$z_collider <- 0.5 * d$rhc_bin + 0.5 * d$death180_bin + rnorm(nrow(d), 0, 0.4)
covs_with_collider <- c(covs_correct, "z_collider")
res_E <- run_methods(d_coll, covs_with_collider, "E. 控制对撞节点 z_collider")

# Anti-pattern F：函数形式误设——把 APACHE 改成二值（< 50 vs >= 50），损失重症区段非线性
covs_binary_apache <- setdiff(covs_correct, "apache_score")
d_bin <- d |> mutate(apache_high = as.integer(apache_score >= 50))
covs_F <- c(covs_binary_apache, "apache_high")
res_F <- run_methods(d_bin, covs_F, "F. APACHE 二值化（误设函数形式）")

results_table <- bind_rows(res_base, res_A, res_B, res_C, res_D, res_E, res_F)
cat("\n错误调整集对比表：\n")
print(results_table, n = Inf, width = Inf)

saveRDS(results_table, "/tmp/rhc_anti_patterns.rds")

cat("\n================ DONE ================\n")
