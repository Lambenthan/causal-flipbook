# 因果推断实践：基于 RHC 数据的 R 语言方法手册

> ICU 里给危重病人插右心导管 RHC，到底是救人还是害人？

1996 年，Alfred Connors 和同事在 *JAMA* 报告了一项观察性研究：5735 名 ICU 危重症患者中，接受右心导管监测的患者 180 天死亡率比未接受者高出 7.5 个百分点。这 7.5 个百分点是 RHC 真实的因果效应吗？还是 RHC 组的患者本来就更重？

全书围绕这一份数据、这一个问题展开。每章引入一种因果推断方法，章末把该方法在 RHC 数据上得到的 ATE 估计填进同一张累积对比表。读到第 10 章，读者就能看到九种方法对同一个因果问题的回答有多接近、有多分歧。第 1–6 章的核心概念配有 ≤ 12 秒的轻量动画，跟着书稿原文一起读。

<div align="center">
  <img src="assets/chap01/01_confounding_dag.gif" alt="适应证混杂 DAG" width="560" />
</div>

---

## 章节目录

全书共 10 章。点击章节标题进入。

<div align="center">

| 章 | 标题 | 方法 | 主要可视化 |
|:--|:--|:--|:--|
| [第 1 章](chapters/chap01.md) | 问题与数据：RHC 数据集与适应证混杂 | 数据探索 + Table 1 | 适应证混杂 DAG · 潜在结果分叉 · 粗死亡率分解 |
| [第 2 章](chapters/chap02.md) | 因果结构与识别条件 | DAG + 三大假设 | 节点类型 · 后门路径 · 可交换性 · 正值性违反 |
| [第 3 章](chapters/chap03.md) | 回归调整：从粗关联到条件关联 | 逐步加变量 | 系数漂移 · 混杂吸收 · 非压缩性 · Table 2 谬误 |
| [第 4 章](chapters/chap04.md) | G 计算：构造反事实人群 | 反事实预测 + bootstrap | 两列填补 · 分层标准化 · bootstrap CI |
| [第 5 章](chapters/chap05.md) | 倾向得分：匹配、加权与平衡诊断 | PSM、IPW、OW | 降维定理 · PSM 配对 · IPW 权重雷区 |
| [第 6 章](chapters/chap06.md) | 双重稳健估计：AIPW 的两路加权 | AIPW 手动实现 | 偏差交叉消除 · AIPW vs 单模型 |
| [第 7 章](chapters/chap07.md) | 机器学习增强：Super Learner、DML 与 TMLE | DML + TMLE | 参数模型天花板 · SL 凸组合 · 5 折交叉拟合 · targeting step · 9 方法收敛 |
| [第 8 章](chapters/chap08.md) | 敏感性分析：未测量混杂的压力测试 | E-value + sensemakr | 未测量混杂 DAG · E-value 曲线 · APACHE 基准 · 稳健等高线 |
| [第 9 章](chapters/chap09.md) | 异质性效应：因果森林与个体化 CATE | grf 因果森林 | ATE vs CATE · 诚实分裂 · CATE 直方图 · 变量重要性 · 子群 CATE |
| [第 10 章](chapters/chap10.md) | 全书汇总：十种方法的终极对比 | 终极对比表 + 结论 | 终极森林图 · 方法决策树 |

</div>

---

## 全书路线图

本书只回答一个问题：**RHC 是否因果地增加了 ICU 患者的 180 天死亡率？** 每一章用一种不同的因果推断方法来回答它，最后汇总比较。同一份数据、同一个问题，九种方法各自从不同角度切进来，读者可以亲手看到每种方法的假设、操作和结论有什么异同。

写作沿用经典分卷结构：

- **第 1 部分 · 问题与因果结构**（第 1–2 章）：把 RHC 数据搬上桌、把假设画成 DAG
- **第 2 部分 · 经典估计与双重稳健**（第 3–6 章）：回归、G 计算、倾向得分、AIPW
- **第 3 部分 · 机器学习增强**（第 7 章）：Super Learner / DML / TMLE
- **第 4 部分 · 稳健性、异质性与汇总**（第 8–10 章）：敏感性分析、CATE、终极对比

每一章保持"概念先行 → 通俗讲解 → 为什么这样设计 → 雷区分析 → R 代码 → 知识地图"的同一节奏。所有数字都来自 R 在 RHC 数据上的真实运行输出（29 协变量、5735 名 ICU 患者），代码可在 [code/](code/) 目录复现。

---

## 累积对比表（截至第 9 章）

每章末更新方法的 ATE 估计，便于跨章对照。完整表格见每章末尾，详细解读见 [第 10 章](chapters/chap10.md)。

<div align="center">

| 章 | 方法 | 估计 | 95% CI |
|:--:|:--|--:|:--:|
| 第 1 章 | 粗差异 | RD 0.0752 | — |
| 第 3 章 | 回归调整 | OR 1.29 | [1.13, 1.46] |
| 第 4 章 | G 计算 | RD 0.0531 | [0.0260, 0.0819] |
| 第 5 章 | PSM | RD 0.0621 | [0.0304, 0.0906] |
| 第 5 章 | IPW | RD 0.0465 | [0.0162, 0.0725] |
| 第 5 章 | OW | RD 0.0533 | [0.0261, 0.0807] |
| 第 6 章 | AIPW | RD 0.0455 | [0.0181, 0.0729] |
| 第 7 章 | DML | RD 0.0445 | [0.0183, 0.0708] |
| 第 7 章 | TMLE | RD 0.0413 | [0.0161, 0.0665] |
| 第 8 章 | E-value（基于 AIPW） | 1.43 | 翻盘需 3× APACHE 强度 |
| 第 9 章 | 因果森林 ATE | RD 0.0442 | (SE 0.0124) |
| 第 9 章 | CATE 范围 | [-0.0398, 0.1306] | 96.65% 患者 > 0 |

</div>

九种估计方向完全一致，RD 落在 0.0413–0.0621 这个 2 个百分点的窄带内。E-value 1.43 量化了未测量混杂的稳健性边界，CATE 异质性诊断显示 RHC 的有害效应在 ICU 患者群体里近乎普适。最终结论：**在 1989–1994 年美国 5 家教学医院的 ICU 危重症患者群体中，RHC 因果地增加了 180 天死亡率，平均效应约 4–5 个百分点。**

---

## 数据与复现

**原始数据**：`/CausalInferenceBook-v2/data/rhc.csv`，5735 行 × 49 列，来自 Connors 1996 *JAMA* 原始研究的整理版。

**最小复现流程**（macOS / Linux，假设已装好 R 4.5+）：

```bash
# 1. 装 R 包（一次性）
Rscript -e 'install.packages(c(
  "tidyverse", "broom", "MatchIt", "WeightIt", "cobalt",
  "SuperLearner", "DoubleML", "mlr3", "mlr3learners",
  "ranger", "glmnet", "tmle", "EValue", "sensemakr", "grf"
))'

# 2. 跑全书 9 种方法的因果估计
Rscript code/run_chap04_to_06.R   # 回归 / G 计算 / PSM / IPW / OW / AIPW
Rscript code/run_chap07_to_09.R   # DML / TMLE / E-value / sensemakr / 因果森林
```

两个脚本统一 `set.seed(2026)`，结果完全可复现。Apple Silicon 启用 vecLib BLAS 后整套约 5–8 分钟跑完；如发现单个 R 命令耗时超过 10 分钟，多半是 BLAS 没启用，参考第 7 章末注释或运行 `Rscript -e 'La_library()'` 检查。

**章节脚本对应表**：

| 脚本 | 产出对象 | RDS 输出 |
|---|---|---|
| [code/run_chap04_to_06.R](code/run_chap04_to_06.R) | 回归 OR、G 计算 RD、PSM、IPW、OW、AIPW | `/tmp/rhc_estimates.rds` |
| [code/run_chap07_to_09.R](code/run_chap07_to_09.R) | DML、TMLE、E-value、sensemakr、CF 因果森林 | `/tmp/rhc_estimates_chap789.rds` |

**完整参考文献**：[references.md](references.md) 列出本书所引用的方法学论文（潜在结果框架、倾向得分、双重稳健、机器学习因果、敏感性分析、因果森林）与 R 包出处。

**写作风格指南**：[`causal-inference-textbook-writer` skill](../../.claude/skills/causal-inference-textbook-writer/SKILL.md)。
