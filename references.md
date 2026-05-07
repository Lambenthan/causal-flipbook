# 参考文献

本书围绕一份数据展开，正文里点到的核心方法学论文与 ICU 临床证据集中列在这里。所有条目按主题分组，按发表年代排序。需要在某个具体方法处展开阅读时，按章节后括号回查正文位置。

## 数据来源与 RHC 临床背景

- Connors AF Jr, Speroff T, Dawson NV, Thomas C, Harrell FE Jr, Wagner D, et al. *The effectiveness of right heart catheterization in the initial care of critically ill patients.* JAMA. 1996; 276(11): 889–897. （原始 RHC 观察研究，本书数据来源）
- Sandham JD, Hull RD, Brant RF, Knox L, Pineo GF, Doig CJ, et al. *A randomized, controlled trial of the use of pulmonary-artery catheters in high-risk surgical patients.* N Engl J Med. 2003; 348(1): 5–14. （PAC-Man 之前的高风险手术 RCT）
- Harvey S, Harrison DA, Singer M, Ashcroft J, Jones CM, Elbourne D, et al. *Assessment of the clinical effectiveness of pulmonary artery catheters in management of patients in intensive care (PAC-Man): a randomised controlled trial.* Lancet. 2005; 366(9484): 472–477. （PAC-Man，与本书结论方向一致的 RCT）
- ESCAPE Investigators and ESCAPE Study Coordinators. *Evaluation study of congestive heart failure and pulmonary artery catheterization effectiveness: the ESCAPE trial.* JAMA. 2005; 294(13): 1625–1633.

## 因果推断框架（第 1–2 章）

- Rubin DB. *Estimating causal effects of treatments in randomized and nonrandomized studies.* J Educ Psychol. 1974; 66(5): 688–701. （潜在结果框架的奠基性论文）
- Holland PW. *Statistics and causal inference.* J Am Stat Assoc. 1986; 81(396): 945–960. （命名"Rubin 因果模型"，提出因果推断的根本问题）
- Pearl J. *Causal diagrams for empirical research.* Biometrika. 1995; 82(4): 669–688. （DAG 与 do-calculus）
- Pearl J. *Causality: Models, Reasoning, and Inference.* 2nd ed. Cambridge University Press, 2009. （DAG 方法的标准教科书）
- Greenland S, Pearl J, Robins JM. *Causal diagrams for epidemiologic research.* Epidemiology. 1999; 10(1): 37–48.

## 倾向得分与回归（第 3–5 章）

- Rosenbaum PR, Rubin DB. *The central role of the propensity score in observational studies for causal effects.* Biometrika. 1983; 70(1): 41–55. （倾向得分的奠基性论文，被引超过三万次）
- Robins JM. *A new approach to causal inference in mortality studies with a sustained exposure period: application to control of the healthy worker survivor effect.* Math Modelling. 1986; 7(9–12): 1393–1512. （G 公式与时变混杂处理框架）
- Austin PC. *An introduction to propensity score methods for reducing the effects of confounding in observational studies.* Multivariate Behav Res. 2011; 46(3): 399–424. （PSM 实践指南，本书第 5 章卡钳取值依据）
- Li F, Morgan KL, Zaslavsky AM. *Balancing covariates via propensity score weighting.* J Am Stat Assoc. 2018; 113(521): 390–400. （重叠权重 / ATO 估计量）

## 双重稳健与机器学习因果（第 6–7 章）

- Robins JM, Rotnitzky A, Zhao LP. *Estimation of regression coefficients when some regressors are not always observed.* J Am Stat Assoc. 1994; 89(427): 846–866. （AIPW 早期形式）
- Robins JM, Rotnitzky A. *Comment on "Inference for semiparametric models: Some questions and an answer."* Statist Sinica. 2001; 11: 920–936. （双重稳健性的术语化讨论）
- Bang H, Robins JM. *Doubly robust estimation in missing data and causal inference models.* Biometrics. 2005; 61(4): 962–973. （AIPW 双重稳健性的实操化）
- van der Laan MJ, Rubin D. *Targeted maximum likelihood learning.* Int J Biostat. 2006; 2(1): Article 11. （TMLE 的奠基性论文）
- van der Laan MJ, Polley EC, Hubbard AE. *Super learner.* Stat Appl Genet Mol Biol. 2007; 6(1): Article 25. （Super Learner 的提出）
- Chernozhukov V, Chetverikov D, Demirer M, Duflo E, Hansen C, Newey W, Robins J. *Double/debiased machine learning for treatment and structural parameters.* Econometrics J. 2018; 21(1): C1–C68. （DML 的奠基性论文）
- Kang JDY, Schafer JL. *Demystifying double robustness: a comparison of alternative strategies for estimating a population mean from incomplete data.* Statistical Science. 2007; 22(4): 523–539.

## 敏感性分析（第 8 章）

- Cornfield J, Haenszel W, Hammond EC, Lilienfeld AM, Shimkin MB, Wynder EL. *Smoking and lung cancer: recent evidence and a discussion of some questions.* J Natl Cancer Inst. 1959; 22(1): 173–203. （未测量混杂量化的最早数学论证）
- Rosenbaum PR. *Observational Studies.* 2nd ed. New York: Springer, 2002. （Rosenbaum Γ 边界）
- VanderWeele TJ, Ding P. *Sensitivity analysis in observational research: introducing the E-value.* Ann Intern Med. 2017; 167(4): 268–274. （E-value 的提出）
- Cinelli C, Hazlett C. *Making sense of sensitivity: extending omitted variable bias.* J R Stat Soc B. 2020; 82(1): 39–67. （sensemakr 的奠基论文）

## 异质性与因果森林（第 9 章）

- Athey S, Imbens GW. *Recursive partitioning for heterogeneous causal effects.* Proc Natl Acad Sci USA. 2016; 113(27): 7353–7360.
- Wager S, Athey S. *Estimation and inference of heterogeneous treatment effects using random forests.* J Am Stat Assoc. 2018; 113(523): 1228–1242.
- Athey S, Tibshirani J, Wager S. *Generalized random forests.* Ann Stat. 2019; 47(2): 1148–1178. （因果森林的最终版本，grf 包对应论文）

## 教科书与综述

- Hernán MA, Robins JM. *Causal Inference: What If.* Boca Raton: Chapman & Hall/CRC, 2020. （流行病学因果推断标准教材，免费版 https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/）
- Imbens GW, Rubin DB. *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction.* Cambridge University Press, 2015. （社会科学与生物医学因果推断教材）
- Angrist JD, Pischke JS. *Mostly Harmless Econometrics.* Princeton University Press, 2009. （计量经济学因果推断教材）

## R 包

- `MatchIt` — Ho DE, Imai K, King G, Stuart EA. *MatchIt: nonparametric preprocessing for parametric causal inference.* J Stat Softw. 2011; 42(8): 1–28.
- `WeightIt` / `cobalt` — Greifer N. CRAN packages.
- `SuperLearner` — Polley EC, LeDell E, Kennedy C, Lendle S, van der Laan M. CRAN package.
- `DoubleML` — Bach P, Chernozhukov V, Kurz MS, Spindler M. *DoubleML—An object-oriented implementation of double machine learning in R.* J Stat Softw. 2024.
- `tmle` — Gruber S, van der Laan MJ. *tmle: an R package for targeted maximum likelihood estimation.* J Stat Softw. 2012; 51(13): 1–35.
- `EValue` — Mathur MB, Ding P, Riddell CA, VanderWeele TJ. CRAN package.
- `sensemakr` — Cinelli C, Ferwerda J, Hazlett C. CRAN package.
- `grf` — Tibshirani J, Athey S, Wager S, Friedberg R, Miner L, Wright M. CRAN package.

---

<div align="center">

[返回目录](README.md)

</div>
