# gpt2-new 项目说明
写在前面：
这是天坑专业研究生转码的第一个项目，由于非计算机科班所以内容和结构上可能并不标准，这个项目的前半部分是参考nanoGPT的内容，并把数据集替换成中文的数据集，后半部分的模型训练+微调是自己结合论文或其他教学视频完成的。
**ipynb 文件说明**
- `ch01.ipynb` 处理文本数据与简单分词示例：加载对话数据，拆分 `user/assistant` 文本，使用 `jieba` 做中文分词与批量处理，展示基础数据清洗与统计。
- `ch02.ipynb` 注意力机制与因果掩码实现：从基础的 `Attention` 到加入因果掩码的 `CausalAttention`，并扩展到 `MultiHeadAttention`，用于理解 Transformer 的核心计算流程与张量形状变换。
- `ch03.ipynb` 124M 规模的简版 GPT 组网：按模块搭建 `Embedding`、`Causal/Multi-Head Attention`、`MLP` 与残差结构，组合为一个最小可运行的 GPT 模型并演示前向计算与数据构造。
- `ch04.ipynb` 小语料端到端训练演示：从在线文本下载与清洗开始，使用 `tiktoken` 编码，构造 `Dataset/DataLoader`，演示训练数据切片、目标延迟与基础训练循环的关键步骤。
- `ch06.ipynb` 中文预训练数据管线：从 THUCNews 文本与中文维基 XML 解析构建 `datasets` 数据集，做清洗、分段、抽样与 `save_to_disk`，为预训练提供 Arrow 格式语料；包含将多源语料合并与打包的示例。
- `ch07.ipynb` 问答微调数据管线：加载 `webQA` 与 `DuReader` 数据，定义 `process_func` 生成 `input_ids/attention_mask/labels`，使用 `tokenizer/bytebpe_zh` 做编码并准备问答微调所需样本。

**data 目录**
- `data/dureader/` 存放 `train.csv/dev.csv/test.csv` 等问答数据；列通常包含问题与参考答案文本，用于监督微调。
- `data/webQA/` 存放 `train.json/dev.json/test.json` 问答样本；以 JSON 结构组织输入与输出字段。
- `data/thu_ds/` 使用 `datasets.save_to_disk` 的 Arrow 数据集目录结构，包含 `data-*.arrow` 分片、`dataset_info.json/state.json` 等元数据；可直接被 `datasets.load_from_disk` 读取。
- `data/wiki_dataset/` 同样为 Arrow 数据集结构，保存处理后的中文维基页面文本及其缓存分片，便于大规模顺序读取与重用。
- `data/tokenized_merged/` 合并后的 问答 数据集分片，用于后续训练阶段的数据读取。
- `data/packed/` 经过管线打包得到的二进制文件：`*.bin` 为连续 token 流，`*.idx` 记录每个块在 `bin` 中的偏移与长度；适配预训练时的高效顺序读取。

**tokenizer 目录**
- `tokenizer/bytebpe_zh/` 包含 `vocab.json` 与 `merges.txt` 两个核心文件，表示 Byte-Level BPE 词汇与合并规则。
- 代码中加载方式：
  - `model.load_tokenizer` 优先使用 `AutoTokenizer`，回退到 `GPT2TokenizerFast(vocab.json/merges.txt)`，并确保存在 `pad` 等特殊符号。
  - `data_build.get_tokenizer` 优先使用 `AutoTokenizer`，回退到 `ByteLevelBPETokenizer` 并在必要时补全 `pad/bos/eos/unk` 等特殊符号。
- 该分词器用于将原始文本编码为 `input_ids`，并在构造 `labels/attention_mask` 时提供必要的特殊符号 ID。

**checkpoints 目录**
- `ckpt_step_*.pt` 训练过程定期保存的检查点，包含：
  - `model`: 模型权重 `state_dict`
  - `optimizer`: 优化器状态，用于完全恢复训练进度
  - `scaler`: AMP 混合精度的缩放器状态
  - `step`: 保存时的全局步数
  - `cfg`: 模型结构配置（`vocab_size/context_length/emb_dim/n_heads/n_layers/dropout/qkv_bias` 等）
- `final_model.pt` 仅保存最终的 `model.state_dict`（常被转存为 CPU 权重），用于推理与下游微调：
  - 生成方式参考 `train_hfds.py` 中的 `save_final_model`
  - 推理示例参考 `train_hfds.py` 的 `qa_generate`，支持 `top_k/top_p/temperature` 等采样策略
  其中，final_model是中文语句预训练后的模型，checkpoints10000和60000是中文预训练过程的保存模型，可以用于检查训练是否有效
  checkpoints1000和2000是后续问答模型微调的过程保存模型
  由于模型本身大小只有124M，最终问答模型的效果表现并不好，未上传最终的问答模型

> 说明对应的代码入口：
> - 数据打包与混合：`data_build.py` 的 `mix_packed_bins`、`build_pack_from_arrow_buckets`、`build_pack_from_arrow_buckets_streaming`
> - 训练与评估：`train_hfds.py` 的 `notebook_train_hfds`、`save_final_model`、`qa_generate`；`train_loop.py` 的命令行训练入口
> - 模型与加载：`model.py` 的 `GPTModel`、`build_lm_dataloader`、`load_tokenizer`、`compute_lm_loss`
