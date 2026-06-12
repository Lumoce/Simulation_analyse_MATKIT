# MatKit Analyse Software

MatKit 是一个面向 VASP 模拟计算后处理的本地分析软件项目。当前版本重点覆盖需求文档中的 SO4/Cu2O 吸附结构提取，同时搭建了可扩展到复杂有机缓蚀剂、表面能、吸附能、掺杂能、差分电荷密度、Bader 电荷和 AI 辅助建议的框架。

## 已实现能力

- SO4/Cu2O 专项分析：自动识别 SO4，判断吸附在 Cu 上的 O，输出吸附 O-S 键长、吸附 O-S-O 夹角、S 到 Cu2O 表面参考层的 z 向距离。
- 通用吸附几何分析：按吸附物化学式、表面元素、表面方向和截断值识别吸附接触。
- 能量分析：表面能、吸附能、掺杂/缺陷形成能。
- 电荷分析：差分电荷密度、平面平均、Bader ACF.dat 解析与比较。
- 结构分析：POSCAR/CONTCAR 信息、键长键角、分子识别、表面层识别、表面弛豫。
- AI 辅助：支持 DeepSeek/OpenAI 兼容 API；无 API key 时保留本地规则建议。

## 快速开始

在项目目录下运行：

```bash
cd /Users/simon/Documents/ANU/multi_agent/Analyse_Software
conda activate analyse
python -m matkit --help
```

运行内置 SO4/Cu2O 示例：

```bash
python -m matkit adsorption so4-cu2o examples/so4_cu2o/POSCAR --recommend --verbose
```

导出结构化结果：

```bash
python -m matkit adsorption so4-cu2o examples/so4_cu2o/POSCAR \
  --json analysis_outputs/so4_result.json \
  --csv analysis_outputs/so4_metrics.csv
```

通用有机缓蚀剂吸附几何分析示例：

```bash
python -m matkit adsorption geometry CONTCAR \
  --formula C6H5N3 \
  --surface-element Cu \
  --surface-cutoff 2.6 \
  --recommend \
  --json analysis_outputs/inhibitor_geometry.json
```

## 表面过剩能自动计算

新版 `energy surface` 默认使用单质库中的元素化学势，而不是用体相能量并按面积归一。计算式为：

```text
E_surface = (E_slab - sum_i n_i * mu_i) / n_surfaces
```

其中 `n_i` 从 slab 的 `POSCAR`/`CONTCAR` 自动读取，`mu_i` 从 `simple_substance_database` 中对应单质计算的 `OUTCAR`、`OSZICAR` 或 `log` 自动读取，并除以该单质 `POSCAR` 中的原子数。

推荐目录：

```text
simple_substance_database/
  Cu/
    POSCAR
    OUTCAR
  O2/
    POSCAR
    OUTCAR

Surface_energy/
  task.1_top/
    POSCAR
    OUTCAR
  task.1_bridge/
    POSCAR
    OUTCAR
```

单个 slab：

```bash
python -m matkit energy surface \
  --slab Surface_energy/task.1_top \
  --reference-db simple_substance_database \
  --n-surfaces 2 \
  --csv analysis_outputs/surface_task1_top.csv
```

批量 slab：

```bash
python -m matkit energy surface-batch Surface_energy \
  --reference-db simple_substance_database \
  --n-surfaces 2 \
  --csv analysis_outputs/surface_energy_batch.csv
```

如果仍需旧的面积归一公式，可以显式提供 `--bulk` 和 `--n-bulk`：

```bash
python -m matkit energy surface \
  --slab slab/OUTCAR \
  --bulk bulk/OUTCAR \
  --n-bulk 2
```

## SO4/Cu2O 指标定义

- 吸附 O：SO4 中 O 原子到所选 Cu 表面层任一 Cu 原子的距离小于 `--surface-cutoff`。
- S-O 键长：只对“吸附 O”和中心 S 输出主要结果；`--verbose` 会显示 SO4 所有 S-O 键长。
- O-S-O 夹角：两个吸附 O 与中心 S 构成的角；若只识别到一个吸附 O，则不输出该角。
- S 到表面距离：`S_z - reference_surface_z`，同时输出绝对值。参考面默认是自动判断出的外侧 Cu 表面层平均 z 坐标。

如果你的 slab 吸附在下表面，可以加：

```bash
--surface-side bottom
```

如果自动判断选错表面，也可以显式指定 `top` 或 `bottom`。

## AI 接口配置

支持以下环境变量：

```bash
export MATKIT_API_KEY="your-api-key"
export MATKIT_BASE_URL="https://api.deepseek.com"
export MATKIT_MODEL="deepseek-chat"
```

也可以在 CLI 中传入：

```bash
python -m matkit ai suggest \
  --context "SO4/Cu2O bidentate adsorption, E_ads=-1.2 eV" \
  --tools "adsorption energy, Bader charge, differential charge density, DOS" \
  --api-key "$MATKIT_API_KEY"
```

## 建议的后续扩展

- 加入 `ase`/`pymatgen` 可选后端，用于更复杂的晶胞周期边界和文件格式转换。
- 增加图形界面或 Web 面板，把 JSON 结果可视化为结构指标表、趋势图和任务建议。
- 为有机缓蚀剂补充 SMARTS/片段识别规则，例如 N/O/S/P 配位原子识别、芳香环平面到表面的距离。
- 和现有主动学习/VASP 工作流打通：分析结果进入本地数据库，再驱动下一轮候选构型推荐。

## 验证

```bash
python -m unittest discover -s tests
```
