# KPFM: Kinematic-Projected Flow Matching
## 完整方法描述

---

## 1. 问题设定

### 1.1 目标

在FlowDock等Flow Matching模型的基础上，引入运动学约束，使得：
1. **推理时**：生成的轨迹严格保持分子几何约束（键长、键角不变）
2. **训练时**：学习在运动学流形切空间内的速度场

### 1.2 符号定义

| 符号 | 维度 | 含义 |
|------|------|------|
| $x$ | $\mathbb{R}^{3N}$ | 原子笛卡尔坐标（$N$个原子） |
| $q$ | $\mathbb{R}^{M}$ | 自由度向量（DOF） |
| $M$ | 标量 | 总自由度数 = 6 + $L$ + $K$ |
| $J(x)$ | $\mathbb{R}^{3N \times M}$ | 雅可比矩阵 |
| $\text{FK}(q)$ | $\mathbb{R}^{3N}$ | 正向运动学函数 |
| $t$ | $[0,1]$ | Flow Matching时间参数 |

### 1.3 DOF向量结构

$$q = \begin{bmatrix} \underbrace{t_x, t_y, t_z}_{\text{配体平移 (3)}} & \underbrace{r_x, r_y, r_z}_{\text{配体旋转 (3)}} & \underbrace{\tau_1, \ldots, \tau_L}_{\text{配体扭转角 (L)}} & \underbrace{\chi_1, \ldots, \chi_K}_{\text{侧链Chi角 (K)}} \end{bmatrix}^T$$

典型值：$L \approx 5\text{-}15$（配体可旋转键），$K \approx 10\text{-}40$（pocket侧链）

---

## 2. 核心数学框架

### 2.1 运动学流形

定义运动学流形 $\mathcal{M} \subset \mathbb{R}^{3N}$：

$$\mathcal{M} = \{x \in \mathbb{R}^{3N} : x = \text{FK}(q), \, q \in \mathcal{Q}\}$$

其中 $\mathcal{Q} = \mathbb{R}^3 \times SO(3) \times \mathbb{T}^{L+K}$ 是DOF空间（$\mathbb{T}$为圆周，表示角度）。

**关键性质**：$\mathcal{M}$ 是 $\mathbb{R}^{3N}$ 的低维子流形，维度为 $M \ll 3N$。

### 2.2 切空间

在点 $x = \text{FK}(q)$ 处的切空间：

$$T_x\mathcal{M} = \text{Im}(J(q)) = \{v \in \mathbb{R}^{3N} : v = J(q) \cdot \dot{q}, \, \dot{q} \in \mathbb{R}^M\}$$

**雅可比矩阵** $J(q) \in \mathbb{R}^{3N \times M}$ 将DOF速度映射到原子速度：

$$\dot{x} = J(q) \cdot \dot{q}$$

### 2.3 正交投影

任意原子空间向量 $v \in \mathbb{R}^{3N}$ 到切空间的正交投影：

$$\text{Proj}_{T_x\mathcal{M}}(v) = J \cdot J^+ \cdot v$$

其中 $J^+ = (J^T J)^{-1} J^T$ 是Moore-Penrose伪逆。

---

## 3. 训练阶段

### 3.1 训练数据生成

给定holo结构 $x_1$ 和对应的DOF $q_1$：

**Step 1**: 采样先验 $q_0 \sim p_{\text{prior}}(q)$
$$p_{\text{prior}}(q) = \mathcal{N}(t; 0, \sigma_t^2 I) \cdot \mathcal{N}(r; 0, \sigma_r^2 I) \cdot \text{Uniform}(\tau; -\pi, \pi) \cdot \text{Uniform}(\chi; -\pi, \pi)$$

推荐值：$\sigma_t = 5\text{Å}$，$\sigma_r = 0.5\text{rad}$

**Step 2**: DOF空间测地线插值
$$q_t = \text{GeodesicInterp}(q_0, q_1, t)$$

其中对于角度DOF：
$$\text{GeodesicInterp}(\theta_0, \theta_1, t) = \theta_0 + t \cdot \text{wrap}(\theta_1 - \theta_0)$$
$$\text{wrap}(\delta) = \text{atan2}(\sin\delta, \cos\delta)$$

**Step 3**: 正向运动学得到 $x_t$
$$x_t = \text{FK}(q_t)$$

### 3.2 训练目标构造（核心）

**Step 4**: 计算DOF空间目标速度
$$\dot{q}_{\text{target}} = \frac{\text{GeodesicDiff}(q_1, q_t)}{1 - t}$$

对于角度DOF：
$$\text{GeodesicDiff}(\theta_1, \theta_t) = \text{wrap}(\theta_1 - \theta_t)$$

**Step 5**: 映射到原子空间（关键步骤！）
$$\boxed{v_{\text{target}} = J(q_t) \cdot \dot{q}_{\text{target}}}$$

**为什么这样做？**
- $v_{\text{target}} \in \text{Im}(J) = T_{x_t}\mathcal{M}$，即目标速度**精确在流形切空间上**
- 网络学习的目标是切空间内的速度，而非任意方向

### 3.3 损失函数

$$\mathcal{L} = \mathbb{E}_{t, q_0, q_1} \left[ \| v_\theta(x_t, t) - v_{\text{target}} \|^2 \right]$$

其中 $v_\theta$ 是神经网络输出的速度预测。

**注意**：这与FlowDock的损失形式兼容，都是原子空间MSE。区别仅在于目标 $v_{\text{target}}$ 的构造方式。

---

## 4. 推理阶段

### 4.1 采样流程

```
输入: 初始坐标 x_1 (噪声), 运动学系统 KinSystem
输出: 去噪后坐标 x_0

初始化:
    q ← ExtractDOF(x_1)  # 从初始坐标提取DOF
    x ← x_1

for t = 1.0 → 0.0, step = -Δt:
    # 1. 神经网络预测
    v_raw ← NetworkPredict(x, t)  # [3N]
    
    # 2. 投影到流形切空间
    J ← BuildJacobian(x, KinSystem)  # [3N, M]
    dq ← SolveLeastSquares(J, v_raw)  # [M]
    
    # 3. DOF空间Euler积分
    q ← q + Δt · dq
    q[角度部分] ← wrap(q[角度部分])  # 角度wrap
    
    # 4. 正向运动学重建坐标
    x ← ForwardKinematics(q, KinSystem)

return x
```

### 4.2 投影层详细算法

```python
def ProjectionLayer(v_raw, x, J, λ=1e-3):
    """
    将原子空间速度投影到流形切空间
    
    输入:
        v_raw: [3N] 网络输出的原子速度
        x: [3N] 当前原子坐标（用于构建J）
        J: [3N, M] 雅可比矩阵
        λ: 阻尼因子
    
    输出:
        dq: [M] DOF速度
        v_proj: [3N] 投影后的原子速度
    """
    
    # Step 1: 列缩放（处理尺度差异）
    col_norms = ||J[:, i]||  for i = 1..M  # [M]
    J_scaled = J / col_norms  # 列归一化
    
    # Step 2: 阻尼最小二乘求解
    # min_dq ||J_scaled · dq - v_raw||² + λ||dq||²
    A = J_scaled^T @ J_scaled + λ·I  # [M, M]
    b = J_scaled^T @ v_raw            # [M]
    dq_scaled = solve(A, b)           # [M]
    
    # Step 3: 反缩放
    dq = dq_scaled / col_norms        # [M]
    
    # Step 4: 投影到切空间
    v_proj = J @ dq                   # [3N]
    
    return dq, v_proj
```

### 4.3 为什么需要列缩放？

不同DOF的尺度差异大：
- 平移：单位Å，典型变化量 ~1Å
- 旋转：单位rad，典型变化量 ~0.1rad
- 对应的Jacobian列的范数差异可达10倍以上

**不缩放的问题**：最小二乘会偏向于"便宜"的DOF（J列范数大的），导致其他DOF响应不足。

**列缩放效果**：所有DOF在优化中被平等对待。

---

## 5. 雅可比矩阵构建

### 5.1 Jacobian结构

$$J = \begin{bmatrix} J_{\text{trans}} & J_{\text{rot}} & J_{\tau_1} & \cdots & J_{\tau_L} & J_{\chi_1} & \cdots & J_{\chi_K} \end{bmatrix}$$

### 5.2 各分量计算

**平移** ($J_{\text{trans}} \in \mathbb{R}^{3N \times 3}$):
$$J_{\text{trans}}[3i:3i+3, :] = I_3 \quad \text{(配体原子)}$$
$$J_{\text{trans}}[3i:3i+3, :] = 0 \quad \text{(蛋白原子)}$$

**旋转** ($J_{\text{rot}} \in \mathbb{R}^{3N \times 3}$):
$$J_{\text{rot}}[3i:3i+3, :] = -[r_i - c]_\times \quad \text{(配体原子)}$$

其中 $c$ 是配体质心，$[v]_\times$ 是向量 $v$ 的反对称矩阵。

**扭转角** ($J_{\tau_k} \in \mathbb{R}^{3N \times 1}$):
$$J_{\tau_k}[3i:3i+3, 0] = \begin{cases} a_k \times (r_i - p_k) & \text{if } i \in \text{downstream}(k) \\ 0 & \text{otherwise} \end{cases}$$

其中 $a_k$ 是扭转轴方向，$p_k$ 是轴上一点。

**Chi角**: 与扭转角类似。

---

## 6. 正向运动学 (FK)

### 6.1 算法

```python
def ForwardKinematics(q, x_ref, KinSystem):
    """
    从DOF重建原子坐标
    
    输入:
        q: [M] DOF向量
        x_ref: [3N] 参考坐标（初始构象）
        KinSystem: 运动学定义
    
    输出:
        x: [3N] 重建的原子坐标
    """
    x = x_ref.copy()
    
    # 解析DOF
    trans = q[0:3]
    rot = q[3:6]  # axis-angle
    torsions = q[6:6+L]
    chis = q[6+L:]
    
    # 1. 配体刚体变换
    lig_coords = x[lig_mask]
    centroid = mean(lig_coords)
    
    # 旋转（绕质心）
    R = Rodrigues(rot)  # axis-angle → 旋转矩阵
    lig_coords = (lig_coords - centroid) @ R.T + centroid
    
    # 平移
    lig_coords = lig_coords + trans
    x[lig_mask] = lig_coords
    
    # 2. 配体扭转角
    for k, tor_def in enumerate(KinSystem.torsions):
        axis_point = x[tor_def.axis_u]
        axis_dir = normalize(x[tor_def.axis_v] - axis_point)
        R_tor = Rodrigues(axis_dir * torsions[k])
        
        for i in tor_def.downstream_atoms:
            x[i] = axis_point + R_tor @ (x[i] - axis_point)
    
    # 3. 侧链Chi角（类似）
    ...
    
    return x
```

### 6.2 Rodrigues旋转公式

$$R = I + \sin\theta \cdot K + (1 - \cos\theta) \cdot K^2$$

其中 $K = [a]_\times$ 是旋转轴 $a$ 的反对称矩阵，$\theta = \|a\|$。

---

## 7. 与FlowDock的关系

### 7.1 兼容性

| 组件 | FlowDock | KPFM |
|------|----------|------|
| 网络结构 | 不变 | 不变 |
| 输入 | $x_t, t$ | $x_t, t$ |
| 输出 | $v \in \mathbb{R}^{3N}$ | $v \in \mathbb{R}^{3N}$ |
| 损失空间 | 原子空间MSE | 原子空间MSE |
| **训练目标** | $v = \frac{x_1 - x_t}{1-t}$ | $v = J \cdot \frac{q_1 - q_t}{1-t}$ |
| **推理** | Euler in $\mathbb{R}^{3N}$ | Euler in DOF + FK |

### 7.2 KPFM的修改点

1. **训练数据生成**：目标速度通过 $J \cdot \dot{q}$ 构造
2. **推理采样循环**：加入投影层 + FK

---

## 8. 实现检查清单

### 训练时
- [ ] 从holo结构提取目标DOF $q_1$（扭转角、Chi角）
- [ ] 采样先验 $q_0$
- [ ] DOF空间测地线插值得 $q_t$
- [ ] FK得 $x_t$
- [ ] 构建 $J(q_t)$
- [ ] 计算 $\dot{q}_{\text{target}}$（注意角度用测地线差）
- [ ] 计算 $v_{\text{target}} = J \cdot \dot{q}_{\text{target}}$
- [ ] 损失 = MSE($v_\theta$, $v_{\text{target}}$)

### 推理时
- [ ] 初始化DOF状态 $q$
- [ ] 循环: 网络预测 → 投影 → DOF积分 → FK
- [ ] 角度DOF需要wrap到 $[-\pi, \pi)$

---

## 9. 超参数建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 阻尼因子 $\lambda$ | $10^{-3}$ | 太大会欠约束，太小可能数值不稳定 |
| 先验平移std | 5Å | 覆盖典型binding pocket尺度 |
| 先验旋转std | 0.5rad (~30°) | 适中的初始旋转扰动 |
| 最大平移步长 | 2Å/step | 防止单步跳太远 |
| 最大旋转步长 | 0.5rad/step | 约30°/step |
| 最大扭转步长 | 0.3rad/step | 约17°/step |
| 采样步数 | 50-100 | 与FlowDock类似 |

---

## 10. 总结

**KPFM的核心思想**：

1. **定义运动学流形** $\mathcal{M}$，由DOF参数化
2. **训练目标在流形切空间上**：$v_{\text{target}} = J \cdot \dot{q}_{\text{target}} \in T_x\mathcal{M}$
3. **推理时投影保证约束**：$\dot{q} = J^+ v_{\text{raw}}$，然后FK重建

**优势**：
- 严格保持键长/键角等几何约束
- 与FlowDock框架完全兼容
- 推理时的投影是额外安全保障


## 计算开销 (Computational Cost)

**问题**：计算 $3N \times M$ 的雅可比矩阵及其伪逆涉及矩阵乘法和求逆。对于大蛋白（$N$ 很大）和多侧链（$M$ 变大），这步操作在每一步 ODE 积分中都要进行。
**分析**：

* $N \approx 2000 \sim 5000$ 原子。
* $M \approx 6 + 10 + 30 \approx 50$ 自由度。
* $J^T J$ 是 $50 \times 50$ 的矩阵，求逆很快。
* 瓶颈在于构建 $J$ 和矩阵相乘。
  **建议**：由于 $J$ 是稀疏的（特定侧链只影响特定下游原子），利用稀疏矩阵运算或针对性的 CUDA Kernel 可以显著加速。但在 Python 原型阶段，建议先在 CPU/GPU 上使用 dense matrix 验证效果，速度应该在可接受范围内。

## 数据预处理的复杂性

**问题**：训练阶段需要 $q_1$（Target DOF）。这意味着您必须对 PDBBind 数据集中的每个晶体结构进行**逆运动学（Inverse Kinematics）**或几何分析，以提取准确的键长、键角、扭转角和刚体位姿。
**建议**：

* 不要尝试实时计算。
* **预处理**：使用 RDKit (针对配体) 和 BioPython/PyRosetta (针对蛋白) 预先提取所有样本的 torsion angles 和 rigid body transform，存为训练数据。
* **拓扑匹配**：确保生成的 ESMFold 结构与晶体结构的原子拓扑一致，否则 DOF 映射会出错。

