# 1D Altermagnetic FFLO TSC 相图的机器学习融合方向

八个可能方向与后续项目化落地方案

说明：本文档保留上一轮答复中的八个方向及其后续推荐路线、最小项目设计和核心判断；仅进行文档化排版，不删减实质内容。

## 总体思路

可以。结合你现在的代码结构，我建议不要直接“用 ML 替代理论计算”，而是先把 ML 放在**相图加速、边界识别、拓扑分类、反向设计、物理发现**这几个位置上。

你的 `eta_phase_diagram_cuda.py` 已经输出 `eta_matrix`, `q_opt_matrix`, `delta_opt_matrix`, `ic_plus_matrix`, `ic_minus_matrix` 等相图数据；`tfflo_1d_cuda.py` 已经有 `JA` 扫描、自由能最小化、`Z2` 判据和 OBC 谱验证，这些都天然适合作为 ML 数据集。

## 方向一：相图 surrogate model，加速参数扫描

这是最直接、最实用的方向。

你现在的核心耗时来自对每个参数点反复扫描 `Delta`, `q`, `k`，再通过 BdG 本征值求和得到自由能最小值。`eta_phase_diagram_cuda.py` 中的 `compute_omega_min_q_batch` 本质上是在每组 `(kT, JA)` 上扫描 `delta_vec` 和 `q_vec`，然后得到 `q_opt`, `delta_opt`。

可以训练一个 surrogate model：

```text
(k_B T, J_A, mu, lambda_Ry, lambda_Rz, U)
    -> (Delta_opt, q_opt, eta, I_c^+, I_c^-)
```

第一版不需要很复杂，MLP / XGBoost / Random Forest 都可以。目标不是追求“神经网络有多高级”，而是证明：

- 用少量精确 BdG 数据训练后，可以快速预测高分辨率相图；
- 在相边界附近仍保持较高准确率；
- 可以把原来几个小时甚至更久的扫描变成秒级预估。

更高级一点，可以不是直接预测 `q_opt`, `Delta_opt`，而是学习自由能面：

```text
(k_B T, J_A, Delta, q) -> Omega(Delta, q)
```

然后用神经网络自由能面做快速最小化。这比直接预测相图更物理，因为它保留了“自由能极小决定相”的结构。

推荐落地顺序：

1. 先做直接回归：

```text
input:  kT, JA
output: delta_opt, q_opt, eta, ic_plus, ic_minus
```

2. 然后扩展到：

```text
input:  kT, JA, mu, lambda_ry, lambda_rz, u
output: delta_opt, q_opt, eta, z2, phase_label
```

3. 最后再做：

```text
input:  kT, JA, delta, q
output: Omega
```

这个方向最适合作为“machine-learning-assisted construction of altermagnetic FFLO topological superconductivity phase diagrams”的第一步。

## 方向二：主动学习 active learning，只在相边界附近精算

你现在的网格已经手动体现了一种物理直觉：`kT` 和 `JA` 都用了非均匀网格，尤其 `JA` 在相变附近加密，默认中心大约在 `JA ~= 0.6`。代码里 `build_ja_vec` 明确把 `JA` 网格在 `ja_refine_center` 附近加密；`build_kt_vec` 也把低温区和中温区分段加密。

ML 可以把这个“人工加密”升级成“自动加密”。

做法是：

1. 先用粗网格计算一版相图；
2. 训练一个分类器或高斯过程模型；
3. 找到模型最不确定的区域，例如：
   - `P(cFFLO) ~= P(tFFLO)`；
   - `P(TSC) ~= 0.5`；
   - `eta` 梯度很大；
   - `Delta_opt` 接近 0；
   - `q_opt` 跳变；
4. 只在这些区域调用精确 BdG CUDA 代码补点；
5. 反复迭代，直到边界收敛。

这对你特别有价值，因为相图中真正需要高精度的地方不是整个二维区域，而是：

- normal / FFLO 边界；
- cFFLO / tFFLO 边界；
- `Z2` 翻转边界；
- `eta` 正负号翻转或极值区域。

这个方向很适合写成一个独立方法章节：active-learning-assisted phase boundary refinement。

## 方向三：相分类器：normal / cFFLO / tFFLO / diode-enhanced region

你可以从现有输出自动构造标签。比如：

```text
normal:       Delta_opt < epsilon
FFLO:         Delta_opt > epsilon and |q_opt| > q_eps
cFFLO/tFFLO: 由 Z2、OBC edge spectrum 或你已有的解析判据区分
diode+/-:     eta > 0 或 eta < 0
strong diode: |eta| > threshold
```

`tfflo_1d_cuda.py` 里已经有 `compute_z2`，通过 `q_opt` 和 `delta_opt` 计算 `Z2` 指标；同时还构造 OBC Hamiltonian 并计算谱，这可以作为拓扑标签的交叉验证。

第一版可以做一个监督分类器：

```text
input: kT, JA, mu, lambda_ry, lambda_rz, U
label: normal / trivial FFLO / topological FFLO
```

更进一步，可以把输入换成更“物理”的特征：

- Fermi points；
- normal-state band gap；
- spin splitting；
- Fermi velocity asymmetry；
- `q_opt`；
- `delta_opt`；
- `Z2` formula 中 `lhs`, `rhs` 的值；
- minimum BdG gap。

这里有一个比较好的研究点：比较“直接参数输入模型”和“物理特征输入模型”的泛化能力。

如果只输入 `(kT, JA, mu, lambda)`，网络可能只是插值；如果输入 Fermi surface asymmetry、gap closing indicator、`Z2` formula terms，它就更像在学习物理机制。

## 方向四：用 CNN / U-Net 做相图图像分割与边界提取

如果你已经生成了大量二维相图，例如不同 `mu`, `lambda_R`, `U` 下的 `eta(kT, JA)`，可以把相图本身当成图像：

```text
input image:
    eta_matrix
    q_opt_matrix
    delta_opt_matrix
    ic_plus_matrix
    ic_minus_matrix

output image:
    phase mask
    boundary mask
    high-eta region
    topological region
```

你现在的 `eta_phase_diagram_cuda.py` 保存的结果天然是矩阵格式，包括 `eta_matrix`, `q_opt_matrix`, `delta_opt_matrix`, `ic_plus_matrix`, `ic_minus_matrix`。

这可以做三类任务：

1. 相图去噪 / 超分辨率：用低分辨率相图预测高分辨率相图。
2. 边界检测：用 U-Net 找 normal-FFLO、cFFLO-tFFLO、topological transition boundary。
3. 相图压缩与分类：用 autoencoder 把一张相图压缩成低维 latent vector，再看不同参数下相图如何演化。

这个方向比较“视觉化”，适合做展示图，但从物理深度上不如 surrogate/free-energy model。

## 方向五：inverse design，反向寻找强 eta 或稳定 tFFLO 区域

你现在已经能算 `eta`，而且代码里通过 `j(q)` 得到正负临界电流，再计算 diode efficiency。`eta_phase_diagram_cuda.py` 里有 `compute_current_from_omega` 和 `find_eta_from_jq`，最后保存 `ic_plus_matrix`, `ic_minus_matrix`, `eta_matrix`。

这就可以做反向设计：

```text
目标:
    maximize |eta|
    或 maximize eta
    或 maximize topological gap while keeping |eta| large

变量:
    JA, kT, mu, lambda_ry, lambda_rz, U
```

方法可以用：

- Bayesian optimization；
- genetic algorithm；
- reinforcement learning；
- differentiable surrogate optimization。

特别值得做的目标函数是多目标的：

```text
max_theta [ |eta(theta)|, Delta_topo(theta), T_c(theta) ]
```

同时加约束：

- `Delta_opt > threshold`；
- `Z2 = nontrivial`；
- `minimum gap > threshold`；
- `q_opt` in physical range。

这比单纯画相图更进一步：不是问“相图长什么样”，而是问“哪里最适合实现拓扑 FFLO diode effect”。

## 方向六：拓扑判据学习：从 BdG Hamiltonian 直接预测 Z2 / edge mode

这是更有“AI for physics”味道的方向。

你已经有两套信息：

- 周期边界下的 BdG Hamiltonian `bdg_hamiltonian_batch`；
- 开边界下的 `build_obc_hamiltonian` 和 OBC spectrum。

可以构造任务：

```text
input:
    H(k) sampled on k-grid
    或者 eigenvalues/eigenvectors on k-grid

output:
    Z2
    edge zero mode existence
    minimum bulk gap
```

模型选择：

- 1D CNN over `k`；
- Transformer over `k`-points；
- MLP over physics features。

但我建议不要一开始就用 raw Hamiltonian。更稳妥的路线是：

```text
H(k)
    -> eigenvalues, gap, Pfaffian-like indicators, Fermi velocity asymmetry
    -> classifier
```

这样更容易解释，也更容易和你已有的解析 `Z2` 判据对照。

这个方向的论文叙事可以是：

> The network is not used to replace the topological invariant, but to identify robust topological regions and flag possible gap-closing boundaries for exact verification.

也就是 ML 负责筛选，严格拓扑判据负责最终确认。

## 方向七：学习自由能 landscape 的“相变形态”：一阶 / 二阶 / 多极小竞争

你现在不仅有最小值，还可以保留完整的：

```text
Omega(delta, q | kT, JA)
```

这非常有价值。因为 FFLO 相变的核心不只是 `q_opt` 变不变，而是自由能面上多个极小值之间的竞争。

可以训练或分析：

```text
input:  Omega(delta, q) landscape
output: phase type
        first-order / second-order transition
        metastable region
        competing minima positions
```

物理上尤其值得关注：

- `q = 0` minimum 和 `q != 0` minimum 的竞争；
- `Delta = 0` 和 `Delta != 0` 的竞争；
- 拓扑 gap closing 前后的自由能面变化；
- `eta` 极值区域是否对应强非对称自由能曲率。

ML 方法可以很简单：

- PCA；
- autoencoder；
- clustering。

比如把每个参数点的 `Omega(Delta,q)` 当成一张二维图，用 autoencoder 压缩成 latent space。然后看 latent space 是否自然分成 normal、cFFLO、tFFLO、metastable 区域。

这个方向比较有可能产生“非平凡发现”，因为它不是只复现标签，而是分析自由能面结构。

## 方向八：多尺度泛化：从 1D clean model 扩展到 disorder / finite-size / real-space

如果你后面想把问题推进到更真实的系统，可以把 ML 用在 real-space disorder 或 finite-size OBC 计算上。

例如：

```text
input:
    disorder configuration
    JA, mu, lambda, U
    system size Nx

output:
    min gap
    edge mode robustness
    eta degradation
    topological/trivial label
```

这时可以用：

- 1D CNN；
- Graph neural network；
- DeepONet / neural operator。

但这个方向成本更高，建议在前面几个方向跑通后再做。

## 我最推荐的三条路线

### 路线 A：最容易出结果

监督学习预测相图量：

```text
(kT, JA, mu, lambda_Ry, lambda_Rz, U)
    -> Delta_opt, q_opt, eta, Ic+, Ic-, Z2
```

优点：最快、最稳、最容易和当前代码对接。

### 路线 B：最适合提高计算效率

active learning 精修相边界：

```text
粗网格 BdG -> ML 不确定性 -> 自动补点 -> 精细相图
```

优点：真正能减少计算量，而且和你现在手动加密网格的逻辑天然衔接。

### 路线 C：最有物理深度

学习自由能 landscape：

```text
Omega(Delta, q | parameters)
    -> minima competition / phase transition / metastability
```

优点：不只是预测标签，而是分析 FFLO、拓扑转变、二极管效应背后的自由能机制。

## 一个可执行的最小项目设计

我建议你第一步做成这个结构：

```text
ml_phase/
    dataset_builder.py
    train_surrogate.py
    train_classifier.py
    active_refine.py
    evaluate.py
    plot_ml_vs_exact.py
```

### 数据字段

```python
X = [
    kT,
    JA,
    mu,
    lambda_ry,
    lambda_rz,
    u,
]

Y_reg = [
    delta_opt,
    q_opt,
    eta,
    ic_plus,
    ic_minus,
]

Y_cls = [
    phase_label,
    z2_label,
    strong_diode_label,
]
```

### phase label 可以先粗定义

```python
if delta_opt < 1e-3:
    phase = "normal"
elif abs(q_opt) < 1e-2:
    phase = "uniform_SC"
elif z2 > 0:
    phase = "trivial_FFLO"
else:
    phase = "topological_FFLO"
```

这里 `z2 > 0` / `z2 < 0` 的具体物理命名要和你自己的约定核对，不要机械套用。

### baseline 模型

先不要上复杂深度学习。第一版：

```text
XGBoost / RandomForest:
    phase classification
    eta regression

MLP:
    delta_opt, q_opt, eta regression

Gaussian Process / ensemble MLP:
    uncertainty + active learning
```

### 评价指标

1. `Delta_opt` RMSE；
2. `q_opt` RMSE；
3. `eta` RMSE；
4. phase classification accuracy；
5. boundary F1 score；
6. topological label accuracy；
7. exact BdG calls reduced by how many 倍。

## 最重要的判断

你现在最有价值的数据不是单纯的“相图图片”，而是每个参数点背后的：

- `Omega(q)`；
- `Omega(Delta, q)`；
- `q_opt`；
- `Delta_opt`；
- `j(q)`；
- `Ic+ / Ic-`；
- `eta`；
- `Z2`；
- OBC spectrum。

所以 ML 不要只做“看图识别相”，而应该尽量学习这些中间物理量。这样最后的工作才不像普通图像分类，而更像：

> Machine-learning-assisted discovery and acceleration of altermagnetic FFLO topological superconducting phase diagrams.

我会优先建议你从 surrogate regression + active learning boundary refinement 开始，等这部分稳定后，再做自由能 landscape autoencoder 和 inverse design。
