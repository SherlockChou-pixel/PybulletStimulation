# 多模态抓取仿真系统

基于 PyBullet 的 Panda 机械臂多模态抓取毕设工程，融合：

- 视觉
- 压力
- 惯性

实现抓取、验证抬升、搬运、放置，以及闭环夹持调节与抓取回正。

---

## 1. 当前能力

- Panda + PyBullet 抓取仿真
- 视觉定位目标物体
- 接触力/压力代理感知
- 末端与物体运动状态 IMU 代理感知
- 多模态融合输出：
  - `grasp_confidence`
  - `slip_risk`
  - `state`
- 闭环控制：
  - 自适应收紧
  - recenter 回正
- 复杂环境实验：
  - `baseline`
  - `edge_reach`
  - `low_friction`
  - `heavy_object`
  - `combined_challenge`
- 批量实验与自动分析

---

## 2. 目录结构

```text
grasp_system/    核心系统代码
scripts/         批量实验与结果分析脚本
main.py          单次 GUI / DIRECT 运行入口
logs/            运行日志与实验输出（已建议忽略上传）
README.md        项目说明
```

---

## 3. 环境依赖

建议 Python 3.9+。

核心依赖：

- `numpy`
- `pybullet`
- `matplotlib`

安装示例：

```bash
pip install numpy pybullet matplotlib
```

---

## 4. 快速开始

### 4.1 默认 GUI 运行

```bash
python main.py
```

### 4.2 低摩擦场景

```bash
python main.py --scenario low_friction
```

### 4.3 组合挑战场景

```bash
python main.py --scenario combined_challenge
```

### 4.4 DIRECT 模式

```bash
python main.py --direct --scenario combined_challenge
```

---

## 5. GUI 调试常用命令

### 边缘场景

```bash
python main.py --scenario edge_reach
```

### 重物体场景

```bash
python main.py --scenario heavy_object
```

### 组合挑战 + 无闭环

```bash
python main.py --scenario combined_challenge --disable-closed-loop
```

### 组合挑战 + 无回正

```bash
python main.py --scenario combined_challenge --disable-recenter
```

更多命令见：

- `GUI调试与批量实验命令清单.md`

---

## 6. 批量实验

### 基线 20 次

```bash
python scripts/batch_experiments.py --runs 20 --scenario baseline
```

### 低摩擦 20 次

```bash
python scripts/batch_experiments.py --runs 20 --scenario low_friction --output logs/round2/low_friction.csv --log-file logs/round2/low_friction.log
```

### 组合挑战 20 次

```bash
python scripts/batch_experiments.py --runs 20 --scenario combined_challenge --output logs/round2/combined_challenge.csv --log-file logs/round2/combined_challenge.log
```

### 无闭环对照

```bash
python scripts/batch_experiments.py --runs 20 --scenario combined_challenge --disable-closed-loop --output logs/round2/combined_challenge_no_closed_loop.csv --log-file logs/round2/combined_challenge_no_closed_loop.log
```

### 无回正对照

```bash
python scripts/batch_experiments.py --runs 20 --scenario combined_challenge --disable-recenter --output logs/round2/combined_challenge_no_recenter.csv --log-file logs/round2/combined_challenge_no_recenter.log
```

---

## 7. 结果分析

批量实验完成后可执行：

```bash
python scripts/analyze_batch_results.py --input logs/batch_experiments.csv
```

默认输出：

- `logs/batch_analysis/summary.md`
- `logs/batch_analysis/batch_overview.png`
- `logs/batch_analysis/quality_trends.png`

---

## 8. 核心模块说明

- `config.py`：系统参数配置
- `simulation.py`：PyBullet 场景与动力学设置
- `vision.py`：视觉定位
- `pressure.py`：压力/接触代理感知
- `imu.py`：惯性代理感知
- `multimodal.py`：多模态融合
- `robot.py`：机械臂运动与抓取姿态候选
- `gripper.py`：夹爪控制
- `workflow.py`：完整抓取流程控制
- `experiment_config.py`：GUI 与批量实验共享场景配置

---

## 9. 当前建议提交到仓库的内容

建议保留：

- 源码
- 脚本
- 文档
- `.gitignore`

建议忽略：

- `__pycache__/`
- `.vscode/`
- `.codex-test/`
- `logs/`

---

## 10. 相关文档

- `系统设计说明.md`
- `毕设论文-完成度与补短板计划.md`
- `实验计划与当前进度记录-2026-04-03.md`
- `GUI调试与批量实验命令清单.md`
