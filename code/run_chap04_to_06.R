#!/usr/bin/env Rscript
# 在 RHC 数据上跑第 4–6 章的因果估计：G 计算 / PSM / IPW / OW / AIPW

suppressPackageStartupMessages({
  library(tidyverse)
  library(MatchIt)
  library(WeightIt)
  library(cobalt)
  library(broom)
  library(here)
})

set.seed(2026)

d <- read_csv(here("data", "rhc.csv"), show_col_types = FALSE) |>
  mutate(death180_bin = if_else(death180 == "Yes", 1L, 0L),
         rhc_bin      = if_else(rhc == 1, 1L, 0L),
         sex_bin      = if_else(sex == "Male", 1L, 0L),
         dnr_bin      = if_else(dnr_status == "Yes", 1L, 0L))

cat("Sample size:", nrow(d), "\n")
cat("RHC = 1:", sum(d$rhc_bin), "  RHC = 0:", sum(d$rhc_bin == 0), "\n")
cat("Crude RD:", round(mean(d$death180_bin[d$rhc_bin == 1]) -
                       mean(d$death180_bin[d$rhc_bin == 0]), 4), "\n\n")

covs <- c("age", "sex_bin", "edu", "das_index", "apache_score",
          "glasgow_coma_score", "blood_pressure", "wbc", "heart_rate",
          "respiratory_rate", "temperature", "albumin", "hematocrit",
          "bilirubin", "creatinine", "weight",
          "cancer", "cardiovascular", "congestive_hf", "dementia",
          "psychiatric", "pulmonary", "renal", "hepatic",
          "gi_bleed", "tumor", "immunosupperssion", "transfer_hx", "mi")

fml_y_full <- as.formula(paste("death180_bin ~ rhc_bin +", paste(covs, collapse = " + ")))
fml_ps     <- as.formula(paste("rhc_bin ~",                paste(covs, collapse = " + ")))

# ---------- 第 3 章 · 回归 ----------
m_reg <- glm(fml_y_full, data = d, family = binomial)
or_reg <- exp(coef(m_reg)["rhc_bin"])
ci_reg <- exp(confint.default(m_reg))["rhc_bin",]
cat(sprintf("[Reg]  OR = %.4f  CI = [%.4f, %.4f]\n", or_reg, ci_reg[1], ci_reg[2]))

# ---------- 第 4 章 · G 计算 ----------
d1 <- d |> mutate(rhc_bin = 1L)
d0 <- d |> mutate(rhc_bin = 0L)
ey1 <- mean(predict(m_reg, newdata = d1, type = "response"))
ey0 <- mean(predict(m_reg, newdata = d0, type = "response"))
rd_g <- ey1 - ey0

# Bootstrap CI
B <- 500
boot_g <- replicate(B, {
  idx <- sample(nrow(d), replace = TRUE)
  fit <- glm(fml_y_full, data = d[idx, ], family = binomial)
  mean(predict(fit, newdata = d1, type = "response")) -
  mean(predict(fit, newdata = d0, type = "response"))
})
ci_g <- quantile(boot_g, c(0.025, 0.975))
cat(sprintf("[G]    RD = %.4f  CI = [%.4f, %.4f]\n", rd_g, ci_g[1], ci_g[2]))

# ---------- 第 5 章 · PS 模型 ----------
ps_mod <- glm(fml_ps, data = d, family = binomial)
d$ps <- predict(ps_mod, type = "response")
cat(sprintf("PS range: [%.4f, %.4f]; treated mean %.4f, control mean %.4f\n",
            min(d$ps), max(d$ps),
            mean(d$ps[d$rhc_bin == 1]), mean(d$ps[d$rhc_bin == 0])))

# ---------- 第 5 章 · PSM ----------
m_match <- matchit(fml_ps, data = d, method = "nearest",
                   distance = "glm", caliper = 0.2, ratio = 1, replace = FALSE)
d_match <- match.data(m_match)
rd_psm <- mean(d_match$death180_bin[d_match$rhc_bin == 1]) -
          mean(d_match$death180_bin[d_match$rhc_bin == 0])

# Bootstrap PSM CI
boot_psm <- replicate(B, {
  idx <- sample(nrow(d), replace = TRUE)
  mm <- tryCatch(
    matchit(fml_ps, data = d[idx, ], method = "nearest",
            distance = "glm", caliper = 0.2, ratio = 1, replace = FALSE),
    error = function(e) NULL)
  if (is.null(mm)) return(NA)
  dm <- match.data(mm)
  mean(dm$death180_bin[dm$rhc_bin == 1]) - mean(dm$death180_bin[dm$rhc_bin == 0])
})
ci_psm <- quantile(boot_psm, c(0.025, 0.975), na.rm = TRUE)
cat(sprintf("[PSM]  RD = %.4f  CI = [%.4f, %.4f]  (matched n = %d)\n",
            rd_psm, ci_psm[1], ci_psm[2], nrow(d_match)))

# ---------- 第 5 章 · IPW (ATE) ----------
w_ate <- weightit(fml_ps, data = d, method = "glm", estimand = "ATE")
d$w_ate <- w_ate$weights
rd_ipw <- weighted.mean(d$death180_bin[d$rhc_bin == 1], d$w_ate[d$rhc_bin == 1]) -
          weighted.mean(d$death180_bin[d$rhc_bin == 0], d$w_ate[d$rhc_bin == 0])

# Bootstrap IPW
boot_ipw <- replicate(B, {
  idx <- sample(nrow(d), replace = TRUE)
  ww <- tryCatch(weightit(fml_ps, data = d[idx, ], method = "glm", estimand = "ATE"),
                 error = function(e) NULL)
  if (is.null(ww)) return(NA)
  dd <- d[idx, ]; dd$w <- ww$weights
  weighted.mean(dd$death180_bin[dd$rhc_bin == 1], dd$w[dd$rhc_bin == 1]) -
  weighted.mean(dd$death180_bin[dd$rhc_bin == 0], dd$w[dd$rhc_bin == 0])
})
ci_ipw <- quantile(boot_ipw, c(0.025, 0.975), na.rm = TRUE)
cat(sprintf("[IPW]  RD = %.4f  CI = [%.4f, %.4f]  (max w = %.2f)\n",
            rd_ipw, ci_ipw[1], ci_ipw[2], max(d$w_ate)))

# ---------- 第 5 章 · OW (ATO) ----------
w_ato <- weightit(fml_ps, data = d, method = "glm", estimand = "ATO")
d$w_ato <- w_ato$weights
rd_ow <- weighted.mean(d$death180_bin[d$rhc_bin == 1], d$w_ato[d$rhc_bin == 1]) -
         weighted.mean(d$death180_bin[d$rhc_bin == 0], d$w_ato[d$rhc_bin == 0])

boot_ow <- replicate(B, {
  idx <- sample(nrow(d), replace = TRUE)
  ww <- tryCatch(weightit(fml_ps, data = d[idx, ], method = "glm", estimand = "ATO"),
                 error = function(e) NULL)
  if (is.null(ww)) return(NA)
  dd <- d[idx, ]; dd$w <- ww$weights
  weighted.mean(dd$death180_bin[dd$rhc_bin == 1], dd$w[dd$rhc_bin == 1]) -
  weighted.mean(dd$death180_bin[dd$rhc_bin == 0], dd$w[dd$rhc_bin == 0])
})
ci_ow <- quantile(boot_ow, c(0.025, 0.975), na.rm = TRUE)
cat(sprintf("[OW]   RD = %.4f  CI = [%.4f, %.4f]\n", rd_ow, ci_ow[1], ci_ow[2]))

# ---------- 第 6 章 · AIPW ----------
m1_pred <- predict(m_reg, newdata = d1, type = "response")
m0_pred <- predict(m_reg, newdata = d0, type = "response")
aipw_ind <- (m1_pred - m0_pred) +
            (d$rhc_bin / d$ps) * (d$death180_bin - m1_pred) -
            ((1 - d$rhc_bin) / (1 - d$ps)) * (d$death180_bin - m0_pred)
rd_aipw <- mean(aipw_ind)
se_aipw <- sd(aipw_ind) / sqrt(nrow(d))
ci_aipw <- c(rd_aipw - 1.96 * se_aipw, rd_aipw + 1.96 * se_aipw)
cat(sprintf("[AIPW] RD = %.4f  SE = %.4f  CI = [%.4f, %.4f]\n",
            rd_aipw, se_aipw, ci_aipw[1], ci_aipw[2]))

cat("\n================ FINAL TABLE ================\n")
cat(sprintf("%-10s %10s %10s %10s\n", "Method", "Estimate", "CI low", "CI high"))
cat(sprintf("%-10s %10.4f %10.4f %10.4f\n", "Reg(OR)",  or_reg, ci_reg[1], ci_reg[2]))
cat(sprintf("%-10s %10.4f %10.4f %10.4f\n", "G",        rd_g,   ci_g[1],   ci_g[2]))
cat(sprintf("%-10s %10.4f %10.4f %10.4f\n", "PSM",      rd_psm, ci_psm[1], ci_psm[2]))
cat(sprintf("%-10s %10.4f %10.4f %10.4f\n", "IPW",      rd_ipw, ci_ipw[1], ci_ipw[2]))
cat(sprintf("%-10s %10.4f %10.4f %10.4f\n", "OW",       rd_ow,  ci_ow[1],  ci_ow[2]))
cat(sprintf("%-10s %10.4f %10.4f %10.4f\n", "AIPW",     rd_aipw, ci_aipw[1], ci_aipw[2]))

# Save for downstream use
saveRDS(list(
  reg_or = or_reg, reg_ci = ci_reg,
  g_rd = rd_g, g_ci = ci_g,
  psm_rd = rd_psm, psm_ci = ci_psm, psm_n = nrow(d_match),
  ipw_rd = rd_ipw, ipw_ci = ci_ipw, max_w = max(d$w_ate),
  ow_rd = rd_ow, ow_ci = ci_ow,
  aipw_rd = rd_aipw, aipw_ci = ci_aipw
), "/tmp/rhc_estimates.rds")
