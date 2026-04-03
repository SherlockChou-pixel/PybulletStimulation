# GUI 调试与批量实验命令清单

更新时间：2026-04-03

## 1. 使用前说明

默认入口：

- 单次 GUI / DIRECT 调试：`main.py`
- 批量实验：`scripts/batch_experiments.py`

复杂场景预设：

- `baseline`
- `edge_reach`
- `low_friction`
- `heavy_object`
- `combined_challenge`

---

## 2. GUI 调试命令清单

### 2.1 基础命令

#### 默认基线场景
```bash
python main.py
```

#### 直接模式（不打开 GUI）
```bash
python main.py --direct
```

#### 指定随机种子
```bash
python main.py --seed 1001
```

---

### 2.2 预设复杂场景

#### 边缘可达场景
```bash
python main.py --scenario edge_reach
```

#### 低摩擦场景
```bash
python main.py --scenario low_friction
```

#### 重物体场景
```bash
python main.py --scenario heavy_object
```

#### 组合挑战场景
```bash
python main.py --scenario combined_challenge
```

---

### 2.3 GUI 对照调试

#### 低摩擦 + 关闭闭环
```bash
python main.py --scenario low_friction --disable-closed-loop
```

#### 低摩擦 + 关闭回正
```bash
python main.py --scenario low_friction --disable-recenter
```

#### 组合挑战 + 关闭闭环
```bash
python main.py --scenario combined_challenge --disable-closed-loop
```

#### 组合挑战 + 关闭回正
```bash
python main.py --scenario combined_challenge --disable-recenter
```

#### 组合挑战 + 仅抓取不放置
```bash
python main.py --scenario combined_challenge --disable-place
```

---

### 2.4 自定义 GUI 环境

#### 固定到工作空间边缘
```bash
python main.py --fixed-object-position 0.72 0.22 0.2
```

#### 自定义随机范围
```bash
python main.py --object-x-range 0.68 0.74 --object-y-range -0.30 0.30
```

#### 自定义低摩擦
```bash
python main.py --object-lateral-friction 0.2 --floor-lateral-friction 0.35
```

#### 自定义重物体
```bash
python main.py --object-mass 0.4
```

#### 低摩擦 + 重物体 + 随机姿态
```bash
python main.py --object-mass 0.4 --object-lateral-friction 0.2 --floor-lateral-friction 0.35 --randomize-object-yaw
```

#### 限制随机偏航范围
```bash
python main.py --randomize-object-yaw --object-yaw-range -1.57 1.57
```

---

### 2.5 推荐优先测试的 GUI 命令

```bash
python main.py
python main.py --scenario edge_reach
python main.py --scenario low_friction
python main.py --scenario heavy_object
python main.py --scenario combined_challenge
python main.py --scenario combined_challenge --disable-closed-loop
python main.py --scenario combined_challenge --disable-recenter
```

---

## 3. 批量实验命令清单

### 3.1 第一轮：基础统计增强

#### 基线 50 次
```bash
python scripts/batch_experiments.py --runs 50 --scenario baseline --output logs/round2/baseline_50.csv --log-file logs/round2/baseline_50.log
```

#### 无闭环 50 次
```bash
python scripts/batch_experiments.py --runs 50 --scenario baseline --disable-closed-loop --output logs/round2/baseline_no_closed_loop_50.csv --log-file logs/round2/baseline_no_closed_loop_50.log
```

#### 无回正 50 次
```bash
python scripts/batch_experiments.py --runs 50 --scenario baseline --disable-recenter --output logs/round2/baseline_no_recenter_50.csv --log-file logs/round2/baseline_no_recenter_50.log
```

---

### 3.2 第二轮：复杂场景实验

#### 边缘可达：基线
```bash
python scripts/batch_experiments.py --runs 20 --scenario edge_reach --output logs/round2/edge_reach.csv --log-file logs/round2/edge_reach.log
```

#### 边缘可达：无闭环
```bash
python scripts/batch_experiments.py --runs 20 --scenario edge_reach --disable-closed-loop --output logs/round2/edge_reach_no_closed_loop.csv --log-file logs/round2/edge_reach_no_closed_loop.log
```

#### 边缘可达：无回正
```bash
python scripts/batch_experiments.py --runs 20 --scenario edge_reach --disable-recenter --output logs/round2/edge_reach_no_recenter.csv --log-file logs/round2/edge_reach_no_recenter.log
```

#### 低摩擦：基线
```bash
python scripts/batch_experiments.py --runs 20 --scenario low_friction --output logs/round2/low_friction.csv --log-file logs/round2/low_friction.log
```

#### 低摩擦：无闭环
```bash
python scripts/batch_experiments.py --runs 20 --scenario low_friction --disable-closed-loop --output logs/round2/low_friction_no_closed_loop.csv --log-file logs/round2/low_friction_no_closed_loop.log
```

#### 低摩擦：无回正
```bash
python scripts/batch_experiments.py --runs 20 --scenario low_friction --disable-recenter --output logs/round2/low_friction_no_recenter.csv --log-file logs/round2/low_friction_no_recenter.log
```

#### 重物体：基线
```bash
python scripts/batch_experiments.py --runs 20 --scenario heavy_object --output logs/round2/heavy_object.csv --log-file logs/round2/heavy_object.log
```

#### 重物体：无闭环
```bash
python scripts/batch_experiments.py --runs 20 --scenario heavy_object --disable-closed-loop --output logs/round2/heavy_object_no_closed_loop.csv --log-file logs/round2/heavy_object_no_closed_loop.log
```

#### 重物体：无回正
```bash
python scripts/batch_experiments.py --runs 20 --scenario heavy_object --disable-recenter --output logs/round2/heavy_object_no_recenter.csv --log-file logs/round2/heavy_object_no_recenter.log
```

#### 组合挑战：基线
```bash
python scripts/batch_experiments.py --runs 20 --scenario combined_challenge --output logs/round2/combined_challenge.csv --log-file logs/round2/combined_challenge.log
```

#### 组合挑战：无闭环
```bash
python scripts/batch_experiments.py --runs 20 --scenario combined_challenge --disable-closed-loop --output logs/round2/combined_challenge_no_closed_loop.csv --log-file logs/round2/combined_challenge_no_closed_loop.log
```

#### 组合挑战：无回正
```bash
python scripts/batch_experiments.py --runs 20 --scenario combined_challenge --disable-recenter --output logs/round2/combined_challenge_no_recenter.csv --log-file logs/round2/combined_challenge_no_recenter.log
```

---

### 3.3 自定义批量实验

#### 自定义低摩擦重物体
```bash
python scripts/batch_experiments.py --runs 20 --object-mass 0.4 --object-lateral-friction 0.2 --floor-lateral-friction 0.35 --randomize-object-yaw --output logs/round2/custom_heavy_low_friction.csv --log-file logs/round2/custom_heavy_low_friction.log
```

#### 固定物体在边缘位置
```bash
python scripts/batch_experiments.py --runs 20 --fixed-object-position 0.72 0.22 0.2 --scenario-tag fixed_edge --output logs/round2/fixed_edge.csv --log-file logs/round2/fixed_edge.log
```

#### 自定义窄工作区边缘范围
```bash
python scripts/batch_experiments.py --runs 20 --object-x-range 0.68 0.74 --object-y-range -0.30 0.30 --scenario-tag custom_edge_band --output logs/round2/custom_edge_band.csv --log-file logs/round2/custom_edge_band.log
```

---

## 4. 推荐的执行顺序

### P0

先跑基础增强：

1. baseline
2. baseline_no_closed_loop
3. baseline_no_recenter

### P1

再跑最可能拉开差距的困难场景：

1. low_friction
2. combined_challenge
3. edge_reach

### P2

最后再补：

1. heavy_object
2. 自定义组合场景

---

## 5. 当前常见报错与处理

### 报错 1：缺少 pybullet

如果运行时看到：

```text
缺少依赖 pybullet，请先安装后再运行 GUI 实验。
```

或：

```text
缺少依赖 pybullet，请先安装后再运行批量实验脚本。
```

说明当前 Python 环境没有安装 `pybullet`。

安装方式：

```bash
pip install pybullet
```

如果你有多个 Python 环境，建议确认运行命令和安装命令使用的是同一个解释器。

### 报错 2：命令行参数不识别

先检查帮助：

```bash
python main.py --help
python scripts/batch_experiments.py --help
```

如果帮助里看不到参数，说明你运行的不是当前工程目录下的最新文件。

