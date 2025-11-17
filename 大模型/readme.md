# 大模型

## 1.资料

[(93 封私信 / 82 条消息) 【LLaMA-Factory 微调教程】LoRA 微调 改变大模型的自我认知 - 知乎](https://zhuanlan.zhihu.com/p/24909312513)

## 2.笔记

### 2.1 大模型微调

模型路径 huggingface

默认或者只选lora

 rope插值加速方式去掉

不要使用量化功能，需要安装auto-gpt、vllm包，这里坑很多。因为很容易有环境的冲突。

只提供数据加载json，格式Alpaca 和 ShareGPT 两种数据格式，分别适用于指令监督微调和多轮对话任务

Alpaca 格式：

\- 适用于单轮任务，如问答、文本生成、摘要、翻译等。
\- 结构简洁，任务导向清晰，适合低成本的指令微调。

ShareGPT 格式：

\- 适用于多轮对话系统的训练，如聊天机器人、客服助手等。
\- 能够捕捉上下文信息和多轮交互，更贴近真实的人机对话场景。

**训练阶段（训练方式）**

Supervised Fine-Tuning （监督微调）

Reward Modeling（奖励建模）

PPO（Proximal Policy Optimization）

DPO （Direct Preference Optimization）

学习率写死 训练轮次按数据集调整选择默认（随时停止）

最大样本数：根据数据集大小和训练需求设置。主要是防止数据量过大导致的内存溢出问题（写死最保险）

梯度：根据显存情况调整

计算类型：这里支持混合精度训练选择（fp16或 bf16）bf16的效果更佳一些。 bf16对某些架构是不支持的，和硬件有关（GPU的架构）。如果你的硬件不支持 BF16，可以选择 FP16 进行混合精度训练。 NVIDIA 4090 支持 BF16运算。我的服务器是 NVIDIA A10 GPU，是基于 Ampere 架构 的 GPU。同样也支持bf16运算。可以先选择bf16，如果不支持会报错，然后再选择fp16就行，直接FP16

截断长度默认

LoRA秩：LoRA秩越大模型越大，默认秩是8

多卡deepseed 单卡禁用

保存间隔为0



lr0 0.0001

lrf 0.0001

weight_decay 0.0001

optimizer AdamW

