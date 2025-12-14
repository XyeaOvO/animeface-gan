# AnimeFace DCGAN (Hydra + Lightning)

最小可跑通的 DCGAN 训练脚手架，针对 AnimeFace128 数据集，集成 Hydra 配置、PyTorch Lightning 训练、wandb 记录、torch-fidelity 评估、jaxtyping/typeguard/einops 形状约束，以及预设 wandb sweep。

## 环境与安装
- 需要 Python 3.11（`.python-version` 已锁定）。
- 安装依赖（含 dev 工具）：
  ```bash
  uv sync --all-extras --dev
  ```
- 激活 venv：`source .venv/bin/activate`（或使用 `uv run ...` 直接执行）。

## 数据集（AnimeFace128）
- 默认路径：`data/animeface128/`。
- 训练入口会自动调用 `modelscope.msdatasets.MsDataset.load('yanghaitao/AnimeFace128')`，如目录为空会自动下载并拷贝到上述路径。
- 手动准备（可选）：自行将 128x128 的 PNG/JPG 图片放入 `data/animeface128/`，保持目录非空即可跳过自动下载。

## 训练入口（Hydra）
- Hydra 输出目录：`outputs/YYYY-MM-DD/HH-mm-ss/`，所有 checkpoint、样例图、wandb 离线缓存都会放在该目录。
- 基础命令（1 epoch 冒烟，CPU 亦可）：
  ```bash
  uv run python -m anime_gan.train \
    trainer.max_epochs=1 trainer.limit_train_batches=2 trainer.limit_val_batches=0 \
    model.sample_every_n_steps=1 model.sample_grid_size=4 \
    dataset.batch_size=8 logger.wandb.mode=offline eval.enabled=false
  ```
- 常用 Hydra 覆盖示例：`trainer.max_epochs=10 dataset.batch_size=64 logger.wandb.mode=online eval.sample_size=512`。
- 冒烟脚本：`./scripts/train_debug.sh`（等价于上面的最小命令）。

## wandb 使用
- 默认 `logger.wandb.mode=offline`，离线缓存位于每次 Hydra 输出目录下的 `wandb/`。
- 切换在线：覆写 `logger.wandb.mode=online`（可配合环境变量 `WANDB_ENTITY`）。
- 离线同步：`wandb sync outputs/<date>/<time>/wandb/offline-run-*`。

## wandb Sweep
- Sweep 配置：`conf/sweeps/dcgan.yaml`（调参 lr/z_dim/feature_maps/batch_size/seed）。
- 一键启动（会创建 sweep 并立即开 agent）：
  ```bash
  ./scripts/sweep.sh conf/sweeps/dcgan.yaml
  ```
- 也可手动：`wandb sweep conf/sweeps/dcgan.yaml` 然后 `wandb agent <sweep-id>`。

## 离线评估（FID/IS）
- 使用 `torch-fidelity` 计算 FID/IS，默认生成 50k 张图：
  ```bash
  uv run scripts/eval.py --checkpoint outputs/<date>/<time>/checkpoints/last.ckpt \
    --data-dir data/animeface128 --num-samples 50000 --batch-size 64 --device cuda
  ```
- 结果会写入 `outputs/eval/metrics.json`（或你指定的 `--output`）。

## 张量形状 SOP（代码落地）
- 关键接口 dataclass：`anime_gan.data.datamodule.ImageBatch`、`anime_gan.models.dcgan.NoiseBatch`，避免 `Dict[str, Tensor]` 传递。
- 形状标注与校验：`Generator.forward`/`Discriminator.forward` 使用 jaxtyping 标注并加 `@typechecked`；训练步在 `anime_gan.lit.dcgan_module.DCGANModule.training_step`。
- 显式维度变换：统一用 `einops.rearrange`（如判别器 logits 展平）；避免隐式 `view/permute`。
- 命名约定：中间变量标注维度后缀（如 `images_bchw`），便于排查。

## 代码风格与 Git 规范
- 代码检查：`ruff` + `ruff-format`（见 `.pre-commit-config.yaml`），推荐 `pre-commit install`。
- 提交约定：采用 Git flow（feature 分支合 PR）+ Conventional Commits（如 `feat: add fid callback`，`fix: handle dataset download`）。

## 目录速览
```
data/                      # 默认数据目录
conf/                      # Hydra 配置 + sweep 配置
outputs/<date>/<time>/     # 每次运行产物（Hydra 控制）
src/anime_gan/             # 代码：数据/模型/Lightning/工具	scripts/                  # 训练冒烟、sweep、评估脚本
```

## 关键命令清单
- 安装：`uv sync --all-extras --dev`
- 冒烟训练：`./scripts/train_debug.sh`
- 正常训练：`uv run python -m anime_gan.train trainer.max_epochs=5 logger.wandb.mode=online`
- 评估：`uv run scripts/eval.py --checkpoint <ckpt>`
- Sweep：`./scripts/sweep.sh`
