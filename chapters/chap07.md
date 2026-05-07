# 第 7 章 · 机器学习增强：Super Learner、DML 与 TMLE

**本章目标**

- 看清前六章所有方法都共享一个隐含前提：所依赖的回归或倾向得分必须用对的函数形式
- 理解 Super Learner 用交叉验证把多个候选算法加权融合，把"选模型"变成"权重学习"
- 理解 DML 用 Neyman 正交化 + 样本分裂消除机器学习的正则化偏倚
- 理解 TMLE 用一步目标化更新把初始估计推到识别参数的方向上
- 在 RHC 5735 名患者上跑 DML 与 TMLE，与第 3–6 章结果横向对比

---

第 6 章用 logistic 回归同时拟合结局模型与处理模型，AIPW 借两个模型的影响函数把它们的偏差互相抵消。AIPW 估出来 RD = 0.0455，95% CI [0.0181, 0.0729]。这套双重稳健性的承诺写得很漂亮：两个模型只要一个对，估计就一致。但有一条没明说——这个承诺要求"对"指的是参数模型层面对得上真实数据生成机制。如果真实数据里 APACHE 与肌酐之间有强交互，logistic 回归只放主效应，结果模型就错了；如果 ICU 医生看到 APACHE 高于某阈值才考虑插管，倾向得分模型只放线性项，处理模型也错了。两个错误同时发生时，AIPW 的"双保险"就同时失灵。

机器学习的卖点是函数形式不需要研究者预先指定。随机森林、梯度提升树、神经网络可以从数据里自己学到非线性关系与高阶交互。把 AIPW 里两个 logistic 回归换成机器学习，理论上能让"两个模型有一个对"的概率显著上升。但替换不是直接把 `glm()` 改成 `ranger()` 就完事——机器学习算法靠正则化降低过拟合风险，正则化在结果模型上会把估计往零的方向收缩，这种"收缩偏倚"会经过 AIPW 的影响函数传到最终因果估计上。Chernozhukov 等 2018 年提出的 Double Machine Learning，简称 DML，用 Neyman 正交化 + 样本分裂解决这个问题。van der Laan 与 Rubin 2006 年提出的 Targeted Maximum Likelihood Estimation，简称 TMLE，用一步受约束的最大似然修正解决另一类问题——AIPW 在有限样本下可能给出超出概率范围的估计。本章把这两个方法串起来讲。

## 7.1 参数模型的天花板

第 3 章到第 6 章都用了同一种结局模型，AIPW 还在它之上叠了一层倾向得分模型，全是 logistic 回归。这个选择有它的好处，模型解析、计算便宜、CI 有公式。代价是函数形式被锁死成线性主效应。29 个协变量，29 个 $\beta$ 系数，每个系数代表"在所有其他变量固定时，该变量上升一个单位带来的 logit 概率变化"。这条假设很严格：它意味着 APACHE 评分从 30 升到 40 与从 80 升到 90 对死亡率的影响幅度相等（在 logit 尺度上），意味着 APACHE 与肌酐之间没有交互效应。

> [!WARNING]
> **雷区 — 函数形式误设的传播**
>
> 如果真实结局生成过程里 APACHE 与肌酐有正向交互——重症叠加肾功能衰竭比单纯重症的死亡率上升更快——而结果模型只放主效应，那么模型对所有"高 APACHE + 高肌酐"个体的死亡率预测都会偏低。G 计算把 5735 个个体的两列预测取均值，被低估的高风险患者群体在两列里都被低估，差值近似抵消，但 ATE 估计仍会有方向不定的偏差。AIPW 通过倾向得分残差校正能纠正一部分，但纠正本身依赖倾向得分模型本身设对——而倾向得分模型也只是一个 logistic 回归，同样可能漏掉 APACHE 与血压的交互。两个模型同时漏同一个交互项，是双重稳健性最经典的失效模式。

机器学习的方法把"函数形式"这个负担从研究者身上转移给算法。随机森林会自动检测交互效应，根据信息增益决定在哪个变量、哪个分割点切一刀；梯度提升树用残差迭代式拟合，可以捕捉非线性阈值效应；glmnet 通过 L1 正则把不重要的协变量系数压成零。它们的共同特点是不需要研究者写出协变量之间的交互项或高次项，模型自己从数据里学。

但这种灵活性是有代价的。所有机器学习算法都依靠正则化或剪枝来控制过拟合，正则化的方向是把估计往零（或往均值、往简单结构）拉。这种"为了防过拟合而主动压低系数"的机制，在预测任务里是好事——预测精度随之提升；在因果推断里是坏事——它会让因果估计也跟着系统性偏低。Chernozhukov 在 2018 年的论文里把这个现象命名为**正则化偏倚**，并证明朴素地把机器学习代入 G 公式或 IPW 会带来 $O_p(n^{-1/4})$ 量级的偏差，比传统方法的 $O_p(n^{-1/2})$ 慢一个数量级，标准的 95% CI 不再有效。

## 7.2 Super Learner：从单模型到集成

机器学习算法琳琅满目，每种算法都有自己擅长的数据特征。随机森林适合捕捉高阶交互、对异常值稳健；glmnet 在协变量数量远超样本量时仍能给出稀疏估计；boosting 在中等样本上的预测精度通常最高；甚至最朴素的均值预测在某些样本极小的层里反而稳定。研究者面对一个新数据时无法事先知道哪个算法最优，传统做法是按经验选一个、或者按某个交叉验证分数选一个，但这种"先选后用"的流程把不确定性藏起来了。

Stack van der Laan 等 2007 年提出的 Super Learner 用另一个角度处理这个选择。它不挑算法，而是把所有候选算法都跑一遍，再用交叉验证学出一组权重，把所有候选算法的预测加权平均。

> [!NOTE]
> **定义 7.1 — Super Learner**
>
> 给定候选学习算法集合 $\{\hat f_1, \hat f_2, \dots, \hat f_K\}$ ，Super Learner 估计为
>
> $$\hat f_{\mathrm{SL}}(L) = \sum_{k=1}^K \hat\alpha_k \hat f_k(L)$$
>
> 其中权重 $\hat\alpha = (\hat\alpha_1, \dots, \hat\alpha_K)$ 通过最小化交叉验证损失得到
>
> $$\hat\alpha = \arg\min_{\alpha \in \Delta} \sum_{i=1}^n L\Big(Y_i, \sum_{k=1}^K \alpha_k \hat f_k^{(-i)}(L_i)\Big),$$
>
> $\Delta$ 是单纯形 $\{\alpha : \alpha_k \ge 0, \sum_k \alpha_k = 1\}$ ， $\hat f_k^{(-i)}$ 是把第 $i$ 个观察从训练集里去掉之后用第 $k$ 个算法拟合的预测器。

Super Learner 的设计思想是把"算法选择"从一次性决定变成一个可学的权重向量。每个候选算法在交叉验证下的预测精度决定它在最终预测里贡献多少。如果随机森林在这份数据上表现最好，它的权重就接近 1；如果几个算法各有优势，最终权重会分散在几个算法上；如果某个算法明显比其他差，它的权重会自动被压成 0。

> [!IMPORTANT]
> **命题 7.1 — Super Learner 的渐近性质**
>
> 在弱条件下，Super Learner 的预测精度渐近不差于候选算法集合中最优的那一个。换句话说，在所有可能的凸组合权重 $\alpha \in \Delta$ 里，交叉验证选出来的 $\hat\alpha$ 在样本量趋于无穷时收敛到最优组合。

这条性质把研究者从"选错算法"的风险里解脱出来——只要把可能有用的候选算法都放进库里，Super Learner 自动找出最好的组合。代价是计算量。每个候选算法要在 $V$ 折交叉验证下拟合 $V$ 次，再在全样本上拟合一次得到最终预测器，整体计算量是单算法的 $V + 1$ 倍。RHC 数据上 5735 个观察、29 个协变量、4 个候选算法、5 折，单次跑完大概几十秒到一两分钟，仍然在可接受范围。

> [!TIP]
> **解读 — Super Learner 在 RHC 上的权重**
>
> 在 RHC 数据的 2000 名子样本上跑 Super Learner（候选 SL.glm + SL.glmnet + SL.ranger + SL.mean，5 折 CV），结果模型的权重大致是 SL.glm 0.48、SL.ranger 0.52、SL.glmnet 与 SL.mean 接近 0。这告诉我们两件事：第一， logistic 回归与随机森林在这份数据上的 CV 预测精度接近，没有哪一个明显占优；第二，glmnet 的 L1 正则在 29 个变量上没找到比未正则的 logistic 回归更好的稀疏解，所以权重被压成 0。SL.mean 是裸均值预测，这里是作为下界基准，权重为 0 说明集成里所有候选算法都比"猜全样本平均"好，这是基本健康指标。

## 7.3 DML：Neyman 正交化与样本分裂

把 Super Learner 替换 AIPW 里的两个 logistic 回归，理论上是把双重稳健性的灵活性提到一个新台阶。但这条路有一个隐藏的陷阱。机器学习算法的预测在训练集上是过拟合的，把过拟合的预测代回 AIPW 的影响函数公式，残差 $Y - \hat Q(A, L)$ 在训练集上人为变小，影响函数的方差被低估，CI 收得太窄。

DML 给出的解法分两步。第一步是 Neyman 正交化，把因果估计的得分函数写成对辅助函数的扰动一阶不敏感的形式；第二步是样本分裂，把"训练辅助函数"和"用辅助函数估计因果"放到不同子样本上做。

> [!NOTE]
> **定义 7.2 — Neyman 正交得分**
>
> 设目标参数为 $\theta$ ，辅助函数（nuisance function）为 $\eta$ ，对应的得分函数 $\psi(W; \theta, \eta)$ 满足 Neyman 正交条件，如果
>
> $$\left.\frac{\partial}{\partial t} E\big[\psi(W; \theta_0, \eta_0 + t(\eta - \eta_0))\big]\right|_{t=0} = 0,$$
>
> 即在真实参数 $(\theta_0, \eta_0)$ 处，得分对辅助函数的小扰动一阶不敏感。

这个条件听起来抽象，落到 ATE 估计上的具体含义是：估计 ATE 用的得分函数对结果模型 $Q(A, L)$ 与倾向得分 $g(L)$ 的小误差不敏感，即便机器学习把 $\hat Q$ 和 $\hat g$ 估得稍偏，ATE 的估计仍然以 $\sqrt n$ 速率收敛到真值。AIPW 的影响函数就是 ATE 估计的 Neyman 正交得分——这也是为什么 AIPW 拥有双重稳健性。第 6 章的 AIPW 公式
$$\widehat{\mathrm{ATE}} = \frac{1}{n}\sum_i \Big[\hat m_1(L_i) - \hat m_0(L_i) + \frac{A_i}{\hat e(L_i)}(Y_i - \hat m_1(L_i)) - \frac{1 - A_i}{1 - \hat e(L_i)}(Y_i - \hat m_0(L_i))\Big]$$
本身就是 Neyman 正交得分的样本均值。DML 的第一步并没有创造新公式，它只是把 AIPW 的得分形式作为出发点，明确指出这是机器学习能合法嵌入的位置。

第二步是样本分裂。把全样本随机分成 $K$ 折，对每一折 $k$ ，用其余 $K - 1$ 折拟合 $\hat Q^{(-k)}$ 和 $\hat g^{(-k)}$ ，再在第 $k$ 折上算 AIPW 个体得分。每一折的得分都不依赖该折自己的数据，过拟合通道被切断；最后把所有折的得分拼起来取均值得到 DML 估计。

> [!NOTE]
> **定义 7.3 — Double Machine Learning（IRM 形式）**
>
> 在 $K$ 折随机分样本下， $\hat\theta_{\mathrm{DML}}$ 是
>
> $$\hat\theta_{\mathrm{DML}} = \frac{1}{n}\sum_{k=1}^K \sum_{i \in I_k} \Big[\hat m^{(-k)}_1(L_i) - \hat m^{(-k)}_0(L_i) + \frac{A_i}{\hat e^{(-k)}(L_i)}(Y_i - \hat m^{(-k)}_1(L_i)) - \frac{1 - A_i}{1 - \hat e^{(-k)}(L_i)}(Y_i - \hat m^{(-k)}_0(L_i))\Big],$$
>
> 其中 $\hat m^{(-k)}, \hat e^{(-k)}$ 都是用第 $k$ 折之外的样本拟合的机器学习预测器， $I_k$ 是第 $k$ 折的样本下标集合。SE 由影响函数 $\psi(W_i; \hat\theta_{\mathrm{DML}}, \hat\eta^{(-k)})$ 的样本方差给出。

DML 的核心承诺是，只要机器学习辅助函数收敛速度满足 $\|\hat Q - Q_0\| \cdot \|\hat g - g_0\| = o_p(n^{-1/2})$ ，ATE 估计仍以参数速率 $\sqrt n$ 收敛，95% CI 仍渐近有效。两个辅助函数自身的速率可以慢到 $n^{-1/4}$ ，乘起来才需要达到 $n^{-1/2}$ 这个门槛。这意味着每个机器学习模型都允许一定程度的偏差，只要两边偏差不强相关，最终 ATE 仍然能被正交化抵消。

> [!TIP]
> **解读 — DML 在 RHC 上的估计**
>
> 用 ranger（500 棵树、min.node.size = 5）作为 $\hat m$ 与 $\hat g$ 的估计器，5 折交叉拟合 DML 给出 RD = 0.0445，SE = 0.0134，95% CI [0.0183, 0.0708]。这个数字与第 6 章 AIPW 的 RD = 0.0455 几乎重合，方向一致、CI 重叠度极高。这条结果说明在 RHC 数据上，把 logistic 回归换成随机森林对 ATE 估计的影响其实有限——ICU 死亡率与 29 个协变量之间的关系大体上仍是线性可加的，没有出现强非线性或强交互让 logistic 回归显著漏掉的信号。但 DML 的价值不在于此处给出了"更好"的数字，而在于给出了一个不依赖参数模型设定的稳健性确认：换了一种完全不同的函数形式，结论稳定。

## 7.4 TMLE：目标化的最大似然更新

DML 与 AIPW 共享同一个估计公式，区别在于 DML 强调样本分裂、AIPW 用全样本。两者都有一个共同的小问题，影响函数公式里有 $A / \hat e(L)$ 这种倒数，当 $\hat e(L)$ 接近 0 或 1 时项数会爆炸；同时公式末尾的 $\hat m_1 + \frac{A}{\hat e}(Y - \hat m_1)$ 在有限样本里没有保证落在 $[0, 1]$ 区间里——结果可能给出负的死亡概率或大于 1 的死亡概率。这在概率尺度上不合规。

TMLE 的做法是从初始 $\hat Q^0(A, L)$ 出发做一步目标化更新。具体做法是构造一个"clever covariate" $H(A, L) = A / \hat e(L) - (1 - A) / (1 - \hat e(L))$ ，把它当一个新协变量加进 $\mathrm{logit}(\hat Q^0)$ 的 logistic 回归里，只估计一个标量参数 $\epsilon$ ，得到更新后的预测
$$\mathrm{logit}(\hat Q^1(A, L)) = \mathrm{logit}(\hat Q^0(A, L)) + \epsilon \cdot H(A, L).$$

> [!NOTE]
> **定义 7.4 — TMLE 估计量**
>
> TMLE 把初始结果模型预测 $\hat Q^0$ 通过 clever covariate 更新到 $\hat Q^1$ ，再把 $\hat Q^1$ 在全人群上做 G 计算式标准化：
>
> $$\hat\theta_{\mathrm{TMLE}} = \frac{1}{n}\sum_{i=1}^n \big[\hat Q^1(1, L_i) - \hat Q^1(0, L_i)\big].$$
>
> 其中 $\hat Q^1(a, L) = \mathrm{expit}\big(\mathrm{logit}(\hat Q^0(a, L)) + \hat\epsilon \cdot H(a, L)\big)$ ， $\hat\epsilon$ 通过把 $H(A, L)$ 作为新协变量加入 $Y$ 对 $\mathrm{logit}(\hat Q^0)$ 的 logistic 回归（offset 形式）拟合得到。

把 logit 写出来再 expit 回去，这个变换的效果是更新永远在 $[0, 1]$ 范围内进行——更新后的 $\hat Q^1$ 永远是合法的概率，标准化得到的 ATE 永远在 $[-1, 1]$ 之间。这是 TMLE 相对 AIPW 最直接的优势。

更深一层，TMLE 拟合 $\hat\epsilon$ 的最大似然方程恰好是 AIPW 的影响函数等于零的条件。这意味着 $\hat Q^1$ 不仅留在概率空间内，还满足"AIPW 残差校正项归零"的目标化要求——所以叫"目标化最大似然估计"。这条性质让 TMLE 与 DML 共享渐近正态性与 $\sqrt n$ 收敛速率，95% CI 仍由影响函数给出。

> [!TIP]
> **解读 — TMLE 在 RHC 上的目标化更新**
>
> 用 ranger 在 5 折交叉拟合下得到初始 $\hat Q^0$ 与 $\hat g$ 。初始 ranger 预测给出的 naive RD 只有 0.0153，远低于 logistic 回归 G 计算的 0.0531，这正是机器学习正则化偏倚的典型症状——ranger 的预测被收缩到了样本均值附近，处理组与对照组的预测差被压平了。Targeting step 的 $\hat\epsilon = 0.0221$ ，把更新后 $\hat Q^1$ 推回到合理位置。最终 TMLE 估计 RD = 0.0413，SE = 0.0129，95% CI [0.0161, 0.0665]。这个数字与 DML 的 0.0445 与 AIPW 的 0.0455 都接近，三种方法在 RHC 上收敛到 0.04–0.05 的窄区间。

> [!IMPORTANT]
> **命题 7.2 — DML 与 TMLE 的等价性**
>
> 在大样本下，DML 与 TMLE 收敛到同一个真值，渐近方差相等，区别仅在有限样本性质。DML 的优势是计算简单、估计公式直接来自影响函数；TMLE 的优势是估计永远落在概率空间内，目标化更新让有限样本的偏差更小。流行病学背景的研究者通常更熟悉 TMLE 的"似然 + 影响函数"叙事，计量经济学背景的研究者更熟悉 DML 的"正交得分 + 样本分裂"叙事，本质是同一件事的两种表述。

## 7.5 在 RHC 数据上跑 DML 与 TMLE

下面的代码用 `DoubleML` 包跑 5 折 IRM-DML，再手算 5 折交叉拟合 TMLE。两套估计共享同一组协变量与同一份处理 / 结局变量，可以直接对比。

```r
set.seed(2026)
library(tidyverse); library(DoubleML); library(mlr3); library(mlr3learners)
library(ranger)

d <- read_csv(here::here("data", "rhc.csv"), show_col_types = FALSE) |>
  mutate(death180_bin = if_else(death180 == "Yes", 1L, 0L),
         rhc_bin      = if_else(rhc == 1, 1L, 0L),
         sex_bin      = if_else(sex == "Male", 1L, 0L),
         cancer       = as.integer(as.factor(cancer)) - 1L)

covs <- c("age", "sex_bin", "edu", "das_index", "apache_score",
          "glasgow_coma_score", "blood_pressure", "wbc", "heart_rate",
          "respiratory_rate", "temperature", "albumin", "hematocrit",
          "bilirubin", "creatinine", "weight", "cancer", "cardiovascular",
          "congestive_hf", "dementia", "psychiatric", "pulmonary",
          "renal", "hepatic", "gi_bleed", "tumor", "immunosupperssion",
          "transfer_hx", "mi")

X <- as.matrix(d[, covs]); A <- d$rhc_bin; Y <- d$death180_bin

# DML：用 DoubleML + ranger
df_dml <- DoubleMLData$new(data.frame(Y = Y, A = A, X),
                            y_col = "Y", d_cols = "A", x_cols = covs)
dml <- DoubleMLIRM$new(
  data = df_dml,
  ml_g = lrn("regr.ranger", num.trees = 500, min.node.size = 5),
  ml_m = lrn("classif.ranger", num.trees = 500, min.node.size = 5,
             predict_type = "prob"),
  n_folds = 5, score = "ATE")
dml$fit()

# TMLE：5 折交叉拟合 + ranger 初始 Q 与 g + targeting step
trim <- function(p) pmin(pmax(p, 0.005), 0.995)
folds <- sample(rep(1:5, length.out = nrow(d)))
Q0_AL <- Q0_1L <- Q0_0L <- ps <- numeric(nrow(d))
for (k in 1:5) {
  tr <- which(folds != k); te <- which(folds == k)
  rfQ <- ranger(Y ~ ., data = data.frame(Y, A, X)[tr, ],
                num.trees = 500, classification = FALSE, seed = 2026 + k)
  Q0_AL[te] <- predict(rfQ, data.frame(A, X)[te, ])$predictions
  d_te1 <- data.frame(A = 1, X)[te, ]; d_te0 <- data.frame(A = 0, X)[te, ]
  Q0_1L[te] <- predict(rfQ, d_te1)$predictions
  Q0_0L[te] <- predict(rfQ, d_te0)$predictions
  rfG <- ranger(factor(A) ~ ., data = data.frame(A, X)[tr, ],
                num.trees = 500, probability = TRUE, seed = 2026 + k)
  ps[te] <- predict(rfG, data.frame(A, X)[te, ])$predictions[, "1"]
}
Q0_AL <- trim(Q0_AL); Q0_1L <- trim(Q0_1L); Q0_0L <- trim(Q0_0L); ps <- trim(ps)

# Clever covariate + targeting
H_AL <- A / ps - (1 - A) / (1 - ps)
H_1L <- 1 / ps;  H_0L <- -1 / (1 - ps)
logit <- function(p) log(p / (1 - p))
expit <- function(x) 1 / (1 + exp(-x))
eps <- coef(glm(Y ~ -1 + offset(logit(Q0_AL)) + H_AL, family = binomial))[1]
Q1_1L <- expit(logit(Q0_1L) + eps * H_1L)
Q1_0L <- expit(logit(Q0_0L) + eps * H_0L)
tmle_rd <- mean(Q1_1L) - mean(Q1_0L)

# 影响函数 SE
Q1_AL <- ifelse(A == 1, Q1_1L, Q1_0L)
IC <- (A / ps - (1 - A) / (1 - ps)) * (Y - Q1_AL) +
      (Q1_1L - Q1_0L) - tmle_rd
tmle_se <- sd(IC) / sqrt(nrow(d))
```

> [!TIP]
> **解读 — RHC 上 DML 与 TMLE 的对比**
>
> DML 估计 RD = 0.0445，95% CI [0.0183, 0.0708]，标准误 0.0134。TMLE 估计 RD = 0.0413，95% CI [0.0161, 0.0665]，标准误 0.0129。两个 ML 方法的点估计相差 0.003，置信区间几乎完全重叠，与 AIPW 的 0.0455 也吻合。把 logistic 回归换成 ranger 没有显著改变 ATE 方向或量级，这是 RHC 数据上的稳健性证据：在这份数据规模与协变量集下，函数形式误设的潜在偏差不大。注意 TMLE targeting epsilon 仅 0.0221 这一项也证实了这一点——如果初始 ranger 预测严重偏离真实条件期望，目标化更新需要的 epsilon 会更大。

## 7.6 ML 嵌入因果推断的几条边界

机器学习不是因果推断的免罪金牌。它解决的只是"参数模型函数形式可能错"这一个问题，对其他三条识别假设没有任何帮助。

> [!WARNING]
> **雷区 — 机器学习无法救赎的失败模式**
>
> 第一条，可交换性。无论用 logistic 回归还是 ranger 估计倾向得分，模型都只能学到"已经测量到"的协变量与处理之间的关系。如果存在一个未测量的混杂变量 U，机器学习对处理是否插管的预测会被 U 影响——但是在数据里看不到 U，模型学不到对 U 的调整。可交换性靠的是研究者通过 DAG 论证调整集足够，机器学习只是把"用调整集做调整"这一步做得更灵活。
>
> 第二条，正值性。倾向得分接近 0 或 1 的子群，机器学习同样会给出极端预测。RHC 数据上 ranger 倾向得分范围 [0.0167, 0.8917]，比 logistic 回归的 [0.0250, 0.9342] 略保守，但极端权重的雷区还在。任何依赖 $1 / \hat e$ 或 $1 / (1 - \hat e)$ 的方法在正值性近乎违反时都会失稳。
>
> 第三条，机器学习自身的样本量需求比参数模型大。RHC 这份 5735 名患者的数据上 ranger 已经能给出稳定结果，但样本量降到几百时，DML 与 TMLE 的 CI 会显著变宽，参数模型反而可能更可靠。机器学习在因果推断里的优势只在中等到大样本场景中显现。

> [!WARNING]
> **雷区 — Super Learner 库的选择不是越多越好**
>
> Super Learner 的理论保证是"渐近不差于库里最优算法"，但有限样本下库里的弱算法仍然会消耗交叉验证预算。常见的失败模式是把 8 个候选算法都放进库，每个算法在 5 折 CV 下跑 5 次，总计 40 次拟合，运行时间膨胀到不可接受。实务建议是在库里放 3–5 个互补算法，覆盖一个线性方法（glm 或 glmnet）、一个非参数方法（ranger 或 xgboost）、一个朴素基线（mean），三类函数形式假设各有一个候选即可。

## 7.7 累积对比表（截至第 7 章）

<div align="center">

| 章 · 方法 | ATE 估计 | 95% CI | 核心假设 | 局限 |
|:--|:--:|:--:|:--|:--|
| 第 1 章 · 粗差异 | RD = 0.0752 | — | 无 | 未调整任何混杂 |
| 第 3 章 · 回归调整 | OR = 1.29 | [1.13, 1.46] | 可交换性 + 正值性 + 模型设定正确 | 函数形式敏感、报条件 OR、非压缩性 |
| 第 4 章 · G 计算 | RD = 0.0531 | [0.0260, 0.0819] | 可交换性 + 正值性 + 结局模型正确 | 单依赖结局模型，错了无补救 |
| 第 5 章 · PSM | RD = 0.0621 | [0.0304, 0.0906] | 可交换性 + 正值性 + 处理模型正确 | 丢弃 33% 样本，目标人群变成匹配人群 |
| 第 5 章 · IPW | RD = 0.0465 | [0.0162, 0.0725] | 可交换性 + 正值性 + 处理模型正确 | 极端权重导致方差膨胀 |
| 第 5 章 · OW | RD = 0.0533 | [0.0261, 0.0807] | 可交换性 + 正值性 + 处理模型正确 | 估计量从 ATE 变成 ATO |
| 第 6 章 · AIPW | RD = 0.0455 | [0.0181, 0.0729] | 两模型有一个正确 + 可交换性 + 正值性 | 两模型同错或正值性严重违反时仍偏 |
| **第 7 章 · DML** | **RD = 0.0445** | **[0.0183, 0.0708]** | **Neyman 正交 + 样本分裂 + 可交换性 + 正值性** | **辅助函数收敛速率需达到 $n^{-1/4}$** |
| **第 7 章 · TMLE** | **RD = 0.0413** | **[0.0161, 0.0665]** | **目标化更新 + Super Learner + 可交换性 + 正值性** | **预测必须截断在 $[\delta, 1-\delta]$ 防止 logit 爆炸** |

</div>

把 DML 与 TMLE 加进对比表后，全 7 个 RD 估计落在 0.0413 到 0.0621 这个 2 个百分点的窄带内，方向完全一致。函数形式从 logistic 主效应换成随机森林并没有把估计带出这个范围。这条横向稳健性是 RHC 数据上"RHC 提高 180 天死亡率约 4–6 个百分点"这条结论最有力的支撑。但所有这些方法都共享一个未被检验的核心前提——可交换性。第 8 章用 E-value 与 sensemakr 做敏感性分析，量化"如果还存在一个未测量的混杂变量，它要多强才能把这个结论翻盘"。

## 本章知识地图

<div align="center">

| 核心概念 | 核心内容 | 常见误解 | 为什么错 |
|:--|:--|:--|:--|
| Super Learner | 多算法集成 + CV 学习权重，凸组合预测 | 候选算法越多越好 | 弱算法消耗 CV 预算反而拖慢方差，库里 3–5 个互补算法即可 |
| 正则化偏倚 | 机器学习把估计往零收缩，朴素嵌入 G 公式或 IPW 使因果估计偏低 | 用 ML 替换参数模型就万事大吉 | ML 的灵活性带来过拟合风险，未经 Neyman 正交化会把过拟合偏差留在 ATE 里 |
| Neyman 正交化 | 让得分函数对辅助函数小扰动一阶不敏感 | AIPW 与 DML 是两件事 | AIPW 影响函数本身就是 Neyman 正交得分，DML 是 AIPW + 样本分裂 |
| 样本分裂 / 交叉拟合 | 训练辅助函数与估计因果在不同子样本上做 | 全样本一起训和估计计算量小、应该更准 | 训练集上的过拟合会让影响函数残差被低估，CI 收缩失效 |
| TMLE targeting step | 把初始 Q 通过一步 logit-空间 logistic 回归更新到 AIPW 影响函数残差归零 | TMLE 是 AIPW 的复杂版本 | 两者大样本等价，TMLE 在概率空间内更新解决了 AIPW 有限样本出界问题 |
| ML 不能解决可交换性 | 模型只能学测量到的协变量，未测量混杂仍是黑箱 | ML 自动找到所有混杂 | ML 拟合的是 $E[Y \mid A, L]$ ，与 $E[Y(a) \mid L]$ 之间靠的是 DAG 推理，不是数据 |

</div>
