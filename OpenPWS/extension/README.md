# GLM phase-sensitivity extension

本目录是在 [Periodic Weak Spots](https://github.com/UltimateJupiter/Periodic-Weak-Spots)
提交 `09c109910ba29b834f54441d96b442ab9e1634ae` 上的最小独立扩展。
作者的 `periodic_weak_spots/`、`configs/`、依赖锁和许可证保持原样。
GLM 数据构造复用作者的几何设计与验证器；推理通过进程内 worker 扩展接入 vLLM，
不修改模型权重、安装的 vLLM 文件或作者源码。

This extension adds GLM-5.3-Flash experiments to the original release. **GLM defaults
to quote-tolerant answer-prefix accuracy**, defined exactly below; the original
answer-prefix metric is also retained. The included DeepSeek results continue to
use the author's metric. The figure intentionally uses short labels; this document
defines the underlying metrics, inputs and limitations.

## 评分规则：默认值及历史口径

GLM 的默认指标名为 `quote_tolerant`，实现严格等价于：

```python
completion.lstrip().lstrip("\"'`“‘«").lstrip().startswith(gold)
```

依次移除开头空白、一段连续的开头引号字符、随后空白，再判断是否以答案开头。
引号集合为 ASCII 双引号、单引号、反引号，以及 `“`、`‘`、`«`。
这不是任意位置搜索、大小写归一化或完整语义匹配：`The answer is gold` 不算匹配
`gold`，`"wrong gold"` 也不算；`"gold"` 算。规则只执行上述三次去前缀操作，
不会循环跳过交错的空白和引号。与作者规则一样，答案后的续写不影响前缀匹配。

`official` 完全复用作者函数：`completion.strip().startswith(gold)`。
新的 GLM 逐条结果中，`correct` 采用默认指标，`scores` 同时保存两个具名指标；
报告的 JSON/CSV 始终保留两者。可用 `--glm-metric official` 切换报告和图表。
DeepSeek 表格与图表始终使用 `official`，因此不要依据三模型图的绝对高度直接排名。

容忍引号的规则最初是在观察到格式错误后添加的诊断指标，**不是原论文的评分规则，
也不是预先注册的指标**。本整理版将它设为 GLM 默认，但未据此重选样本。
历史原生初次运行得到 84.57%；这里采用与干预相同运行设置的原生重跑结果 84.03%。

## 方法对应关系

| CLI 模式 | 图例 | 索引选择 |
|---|---|---|
| `native` | block-indexer（原生） | 原生选择 512 个完整四-token 池，并保留原生未完成池尾部处理 |
| `hisa` | HISA | 原生选择 2,048 个完整池（8,192 tokens），加最多 3 个未完成池 token；按原始逐-token 索引分数重排，最终选择 2,048 tokens |
| `dsa` | DSA | 对全部因果历史 token 的原始索引 K 打分，直接选择 2,048 tokens |

HISA / DSA 是本仓库对两种推理干预的简称；没有训练新权重。保留原模型的索引 Q/K
投影、head weights、RoPE 和 FP8/Hadamard 约定。两种干预共用同一个原生 FP8 MQA
打分内核。HISA 参考实现先计算额外分数，再屏蔽候选外位置；**这不是候选专用的
性能优化内核，不能由其耗时评价 HISA 的部署速度**。

干预作用于所有 11 个稀疏层。GLM 还有 34 个线性注意力层；该模型的四-token
索引池化不能直接等同于 DeepSeek 的 KV 压缩。原生对照也分配相同实验缓冲区，
但实际推理委托给未修改的原生算子。原生尾部可能增加最多 3 tokens；HISA/DSA
的最终预算严格为 2,048 tokens。

实现限定单序列、eager、无 prefix cache、无 speculative decoding、无 context
parallel。运行时检查连续位置、每请求缓存重置、所有层覆盖、因果性及候选范围。
不要用 `python -O` 关闭这些断言。
首次 GPU 运行会编译缺失的 FlashInfer/CUDA 内核，初始化可能较慢，后续复用编译缓存。

## 已完成的结果

历史真实推理记录位于 `reference/*.jsonl.gz`，各 2,048 条；保留 sample ID、相位、
答案、原始 completion 和输出 token IDs，不包含完整 128K prompts、本地路径、
权重或运行日志。所有分数均可从这些记录重算，文件 SHA256 及原始结果 SHA256
保存在 `reference/provenance.json`。这些是精简前的完整运行结果，**不是精简代码
重新完成的全量实验**。

| 模型 / 方法 | 默认准确率 | 相位最大差距（百分点） | GLM 原始评分 |
|---|---:|---:|---:|
| DeepSeek-V4-Flash-0731 | 90.19% | 20.31 | — |
| DeepSeek-V4.1-Flash | 91.55% | 5.47 | — |
| GLM block-indexer（原生） | 84.03% | 5.86 | 76.81% |
| GLM HISA | 86.33% | 1.95 | 73.19% |
| GLM DSA | 86.38% | 2.73 | 76.22% |

GLM 原生的相位 3、7 较弱；两种干预的观测波动更小。但原运行按 binding 配对
bootstrap 的最大相位差距缩小区间跨零，不能宣称已完全消除相位敏感性。
HISA 的原始评分还出现下降，默认指标提高不能替代这一事实。

## 直接重画结果（不需要 GPU 或模型）

在仓库根目录执行：

```bash
uv run --no-project --python 3.12 --with 'matplotlib>=3.9,<4' python -m extension.report
```

默认生成英文图例，避免依赖操作系统字体。中文图与当前展示版本一致，显式提供字体：

```bash
python -m extension.report --font /path/to/NotoSansCJK-Regular.ttc
python -m extension.report --glm-metric official --output outputs/official
```

输出 `outputs/phase_comparison.{png,pdf,svg}`、`phase_accuracy.csv`、`metrics.json`
和 `RESULTS.md`。三模型共用纵轴；GLM 只有原生、HISA、DSA 三条线。

## 安装与运行 GLM

本扩展的数据构造、推理和报告共用一个 Python 3.12 / vLLM 0.31.0 环境。
作者原始任务所需的 Python 3.11 锁定环境仍按上游 README 单独安装。
已验证的推理栈为 torch 2.13.0+cu130、transformers 5.17.0、FlashInfer 0.7.0.post1，
8 张 NVIDIA L20Z（每卡约 80 GiB）。worker 接口绑定 vLLM 0.31.0；其他版本需要
重新验证。模型权重请遵守模型自身许可。

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python -e . -r extension/requirements.txt
```

使用自己的本地模型目录（须包含原始权重与 tokenizer 文件）：

```bash
export GLM_MODEL=/path/to/GLM-5.3-Flash
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
.venv-vllm/bin/python -m extension.prepare \
  --model "$GLM_MODEL" --filler /path/to/filler.txt --output data/glm
.venv-vllm/bin/python -m extension.score \
  --model "$GLM_MODEL" --dataset data/glm --output results/glm
.venv-vllm/bin/python -m extension.report --glm-results results/glm
```

默认每相位 256 条，共 2,048 条，256 组 binding 全部覆盖八相位。
构造新数据可用 `--rows-per-phase 2560` 扩大为 20,480 条；不能依表现挑选样本。
已有数据可加 `--validate-only` 重新验证。`--limit 8` 仅用于八相位流程检查，
不是相位效应结论；正式运行去掉它即可。恢复运行使用相同命令，身份记录
会校验数据、tokenizer、配置/权重索引、运行源码和版本，拒绝混入其他运行结果。
模型校验记录的是配置、tokenizer 和权重索引的哈希，**不是所有权重分片的内容哈希**；
请保持模型目录只读。并行启动其他 GPU 作业会影响可用显存，运行时请独占所需 GPU。

三种模式在同一驻留引擎中交替执行，各自重新提交完整 prompt；开始前对首条 128K
prompt 做两轮三模式重放，检查每种模式输出 token IDs 完全一致，再写正式观测。
`--variants native` 可单独运行原生模式；后续合并报告要求三组 sample IDs 一致。

所有模式使用相同 FlashInfer 策略设置。默认使用当前硬件的原生 autotune。
若复用原实验的 L20Z 环境，可显式追加：

```bash
--autotune-config extension/reference/l20z_autotune.json
```

该表是硬件/版本专用的运行产物，某些形状仍采用 heuristic fallback；不要在其他
GPU 上盲用。原实验中，HISA/DSA 配对运行，原生对照在单独引擎中采用同一表和缓冲区
设置重跑；精简版将三者统一到同一个入口和引擎，所以需要重新验证运行等价性。

## 输入适配和复现边界

GLM 原查询格式只产生 4,675 个上下文稳定候选，少于互不重叠的 16,000 个 key 和
16,000 个 value 所需的 32,000 个 token。对查询中的两次 key 加双引号后，原 tokenizer
得到 58,695 个候选。仅替换 tokenizer 身份及查询的三个字符串常量；其余几何和
独立验证继续由作者代码执行：16,000 条四-token `key:value。` 记录、目标记录
编号 248（从零开始）、目标平均位置 64K、总长严格 128,000 tokens、平衡相位设计。
GLM 查询为 35 tokens，尾部据此调整。每次验证还按相位抽取一条对照 GigaToken 与 HF。

输入为原始文本，不套 chat template，greedy 生成最多 8 tokens。GLM 相位使用实际
token 源记录起始位置 mod 8；DeepSeek 保留作者数据的源相位标签，默认前置特殊 token
的位置偏移保存在历史记录中。不同 tokenizer 的模型输入文本并不相同。

公开的 filler 来源是
[LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
提交 `021385d68d3202e37893e9d3cd29011c569abe30` 中的
`needlehaystack/PaulGrahamEssays/*.txt`，按文件名排序、直接拼接文本，不插入分隔符。
原实验本地 filler SHA256 为
`b3a70ebc054f2eab5057baf3c4b7e857711472be8086240a516fd29b648ad857`，
与论文的 `51eaf041f492d8fdfdcf87be8756d23888eda0d3cab6816b3c3f11e7682de337`
不同。filler 本文不重新分发。可在仓库外取得该来源，然后：

```python
from pathlib import Path

folder = Path("/path/to/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays")
Path("filler.txt").write_text("".join(p.read_text() for p in sorted(folder.glob("*.txt"))))
```

DeepSeek 可直接沿用作者 CLI，报告也接受其原始 JSON 输出：

```bash
.venv-vllm/bin/python -m periodic_weak_spots.niah_128k build \
  --config configs/niah_128k/evaluation.json --filler-file /path/to/filler.txt \
  --output data/deepseek
.venv-vllm/bin/python -m periodic_weak_spots.niah_128k validate --dataset data/deepseek
.venv-vllm/bin/python -m periodic_weak_spots.niah_128k score --dataset data/deepseek \
  --model /path/to/DeepSeek-V4-Flash-0731 --tensor-parallel-size 8 --results results/deepseek/v4.json
.venv-vllm/bin/python -m periodic_weak_spots.niah_128k score --dataset data/deepseek \
  --model /path/to/DeepSeek-V4.1-Flash --tensor-parallel-size 8 --results results/deepseek/v41.json
.venv-vllm/bin/python -m extension.report --glm-results results/glm \
  --deepseek-results results/deepseek --deepseek-dataset data/deepseek
```

上述作者 CLI 使用自己的默认推理设置，不保证复现本地历史运行的数值。本地历史运行采用
TP8/EP、eager、无 prefix cache、单序列、8,192-token prefill chunks、显存比例 0.92、
seed 33377335。论文 V4.1 每相位 2,560 条，这里的本地历史结果每相位只有 256 条。
因此这里是行为现象复现，不是字节一致的原文复现，也不是全论文实验的完成声明。

## 文件和验证

| 文件 | 职责 |
|---|---|
| `prepare.py` | 局部查询/tokenizer 适配，调用作者构造器和验证器 |
| `score.py` | 三模式推理、恢复校验、实际 token 几何检查与双指标记录 |
| `worker.py` / `patch.py` / `kernels.py` | worker 接入、索引选择和共享原生 FP8 打分 |
| `utils.py` | 两个明确命名的评分规则及少量 I/O |
| `report.py` | 从逐条输出重算、导出表格及三模型图 |
| `test_*.py` | 评分边界/历史结果回归、GPU 独立参考及缓存验证 |
| `reference/` | 五组历史观测、论文数值、输入身份和可选 L20Z 策略表 |

```bash
.venv-vllm/bin/python -m extension.test_cpu
.venv-vllm/bin/python -m extension.test_kernels
.venv-vllm/bin/python -m extension.test_runtime
uv run --no-project --with ruff ruff check extension
```

GPU 测试将原生 FP8 分数与独立反量化 FP32 矩阵乘比较（atol 0.002、rtol 0.0003），
检查候选 gather 与同一原生稠密分数逐位一致、因果 mask、唯一性、top-k 阈值，
以及缓存重置/单 token decode。小规模真实模型检查和本次整理的验证情况见
[VALIDATION.md](VALIDATION.md)。扩展使用仓库的 MIT 许可；上游作者署名和引文保留。
