<div align="center">

# Awesome MetaX

**沐曦 GPU 生态资源清单**

软硬件栈 · 框架适配 · 算子库 · 工具链 · 社区项目

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License: CC0-1.0](https://img.shields.io/badge/License-CC0%201.0-lightgrey.svg)](LICENSE)

</div>

> **关键词**：沐曦 · MetaX · MACA · MXMACA · 曦云 · 曦云 C500 · 曦云 C550 · 曦云 C588 · 国产 GPU · 国产算力 · CUDA 兼容层 · 异构计算 · GPU 算力基准 · GPU 烤机

面向在沐曦（MetaX）曦云系列 GPU 上做训练、推理、算子开发与集群运维的工程师。
收录官方软件栈、框架移植、算子库、调试与基准工具，以及社区项目。

清单持续更新，欢迎 [提交 PR](CONTRIBUTING.md) 补充遗漏项目。

---

## 目录

- [官方资源](#官方资源)
- [软件栈与框架适配](#软件栈与框架适配)
- [算子与通信库](#算子与通信库)
- [开发与调试](#开发与调试)
- [性能基准与压测](#性能基准与压测)
- [迁移与兼容](#迁移与兼容)
- [监控与可观测](#监控与可观测)
- [文档与知识](#文档与知识)
- [上游已支持沐曦的项目](#上游已支持沐曦的项目)
- [平台速查](#平台速查)
- [常见问题](#常见问题)
- [贡献](#贡献)

---

## 官方资源

- **[沐曦开发者社区](https://developer.metax-tech.com/)** — 官方门户。软件栈下载、DeepSeek 专区、算力在线、编程资源、解决方案与运维支持。
- **[MetaX-MACA（GitHub 组织）](https://github.com/MetaX-MACA)** — 官方开源组织，维护 PyTorch / vLLM / TVM / Triton / FlashAttention / NCCL / CUTLASS 等框架的沐曦适配分支。
- **[maca-samples](https://github.com/MetaX-MACA/maca-samples)** — 基础 GPU 编程模式的参考实现，学习 MXMACA 编程模型的起点。
- **[MetaX-MACA.github.io](https://github.com/MetaX-MACA/MetaX-MACA.github.io)** — 组织主页源码。

## 软件栈与框架适配

- **[mcPytorch](https://github.com/MetaX-MACA/mcPytorch)** `16★` — PyTorch 的沐曦适配版本，提供 `torch.cuda` 兼容接口与定制算子。
- **[vLLM-metax](https://github.com/MetaX-MACA/vLLM-metax)** `192★` — vLLM 的沐曦硬件后端插件，社区维护，支持主流 Transformer 模型的高吞吐推理。
- **[mcTVM](https://github.com/MetaX-MACA/mcTVM)** `11★` — 基于 TVM 的深度学习编译器栈，覆盖 CPU / GPU / 专用加速器。
- **[mcTriton](https://github.com/MetaX-MACA/mcTriton)** `3★` — Triton 的沐曦后端，可用 Python DSL 编写自定义 kernel。
- **[flashattn](https://github.com/MetaX-MACA/flashattn)** `5★` — FlashAttention 的沐曦适配版本。
- **[mini-sglang-metax](https://github.com/sxvvv/mini-sglang-metax)** — Mini-SGLang 的沐曦移植，以正确性优先的教学式实现。

## 算子与通信库

- **[mcoplib](https://github.com/MetaX-MACA/mcoplib)** `15★` — 面向沐曦 HD 平台的高性能算子库。
- **[TileOPs-Metax](https://github.com/MetaX-MACA/TileOPs-Metax)** `7★` — 基于 TileLang 的高性能 LLM 算子库。
- **[TileKernels-Metax](https://github.com/MetaX-MACA/TileKernels-Metax)** `4★` — TileLang kernel 集合。
- **[mcTlass](https://github.com/MetaX-MACA/mcTlass)** `2★` — CUTLASS 的沐曦适配，模板化高性能 GEMM 组件。
- **[mccl-nccl](https://github.com/MetaX-MACA/mccl-nccl)** `5★` — NCCL 的沐曦实现，多卡集合通信。
- **[muxi_native_layout_kernels](https://github.com/QingCheng-AI/muxi_native_layout_kernels)** `4★` — 利用 native layout 优化的 GeMV、窄 GEMM、fused MoE 算子。

## 开发与调试

- **[univis](https://github.com/cookiesheep/univis)** `1★` — 逐层数值对照：模型搬到曦云 C500 后算得对不对、从哪一层开始不对，`pip install univis`。
- **[metax-kernelwiki](https://github.com/qhy991/metax-kernelwiki)** `1★` — 曦云 C550 kernel 知识与 MACA 微基准数据，附离线分析。
- **[metax-docs-mcp](https://github.com/AIIanZ/metax-docs-mcp)** — 本地 MCP 与 CLI，检索沐曦官方文档、MetaX-MACA 仓库与 Docker 镜像信息。
- **[metarag-vllm](https://github.com/Json-wu/metarag-vllm)** `1★` — 面向曦云 C500 集群优化的 RAG 系统。

## 性能基准与压测

- **[mx-burner](https://github.com/stlin256/mx-burner)** `1★` — 算力基准与烤机工具，覆盖 FP64 / FP32 / TF32 / FP16 / BF16 / INT8 及 2:4 稀疏，含运行时矩阵尺寸自动探测、HBM / PCIe / P2P 带宽测试。支持 NVIDIA CUDA 与沐曦 MACA 两条后端，附 C550 实测数据。
- **[CUDABurner](https://github.com/stlin256/CUDABurner)** `12★` — mx-burner 的 CUDA 版本，面向 NVIDIA GPU 的同类基准与烤机工具。
- **[shoc-metax](https://github.com/LABELNET/shoc-metax)** — SHOC 基准套件，兼容 MACA 2.27.0.11，可按说明编译运行。

## 迁移与兼容

- **[MXReady](https://github.com/shuljyli/MXReady)** `9★` — 面向 PyTorch CUDA 扩展的沐曦 MXMACA 迁移准备度静态体检工具，迁移前先扫一遍哪些算子会卡住。
- **[metax-model-preflight](https://github.com/xzh25/metax-model-preflight)** — GPTQ / AWQ 量化 checkpoint 在 vLLM 张量并行下的静态预检。
- **[moark-metax-l1-certification](https://github.com/Asudual/moark-metax-l1-certification)** — 沐曦 L1 认证中文实操指南：GPU 环境搭建、模型推理、应用迁移。

## 监控与可观测

- **[metax-mxtop](https://github.com/onlylrs/metax-mxtop)** `20★` — nvitop 风格的终端监控工具，实时查看沐曦 GPU 利用率、显存、功耗与进程占用。

## 文档与知识

- **[metax-kernelwiki](https://github.com/qhy991/metax-kernelwiki)** — kernel 级知识整理，带证据出处与微基准。
- **[metax-docs-mcp](https://github.com/AIIanZ/metax-docs-mcp)** — 把官方文档接进 AI 工作流（MCP / CLI）。
- **[沐曦开发者社区 · 编程资源](https://developer.metax-tech.com/)** — 官方文档、教程与软件栈版本匹配工具。

## 上游已支持沐曦的项目

以下为上游社区自身已加入沐曦后端支持的项目，非沐曦专属：

- **[TileLang](https://github.com/tile-ai/tilelang)** `8.9k★` — 面向高性能 kernel 开发的 DSL，已包含 MXMACA 后端。
- **[FlagCX](https://github.com/flagos-ai/FlagCX)** `262★` — 可扩展的跨芯片通信库，覆盖多种国产加速器。

---

## 平台速查

曦云 C550 单卡公开实测数据（来源：[mx-burner · C550 实测报告](https://github.com/stlin256/mx-burner/blob/main/docs/C550_RESULTS.md)）：

| 项目 | 数值 |
|---|---|
| 显存 | 65536 MiB |
| 计算能力 | 8.0（104 个多处理器） |
| FP16（矩阵乘） | 277.9 TFLOPS |
| BF16（矩阵乘） | 277.7 TFLOPS |
| TF32（矩阵乘） | 129.1 TFLOPS |
| INT8（矩阵乘） | 397.6 TOPS |
| FP32 | 36.3 TFLOPS |
| FP64 | 4.8 TFLOPS |
| HBM 拷贝带宽 | ~960 GB/s |
| 单卡功耗 | 待机 93 W / 满载 451 W（上限 450 W） |

> 上述为单卡独占、矩阵尺寸自动探测后的稳态值。八卡同时满载时因功耗墙耦合，FP16 约 263 TFLOPS。

## 常见问题

**沐曦 GPU 能直接跑 CUDA 程序吗？**

能，但不是二进制兼容。沐曦提供 `cu-bridge` 兼容层，把 CUDA 源码用其编译器（`cucc`）重新编译后运行，链接的是 `libmcblas` / `libmxsml` 等原生库，不是 NVIDIA 驱动。

**沐曦的软件栈叫什么？**

MACA（MXMACA），包含驱动、编译器、数学库（mcBLAS / mcDNN / mcFFT / mcSPARSE）、集合通信（mccl）与性能分析工具。

**迁移一个 CUDA 项目要做哪些事？**

三步：用 `cucc` 替换 `nvcc`；补上兼容层缺失的头文件或常量别名；把 NVML 监控调用指向 `libmxsml`。算子代码通常不用改，但依赖特定 CUDA 特性的部分（如 FP8 / FP4 张量核心指令）需要按平台能力降级。

**曦云 C550 不支持哪些 CUDA 特性？**

公开实测：`cublasGemmEx` 接受 FP64 / FP32 / TF32 / BF16 / FP16 / INT8，拒绝 FP8 / FP6 / FP4 / INT4 及 `COMPUTE_32F_FAST_16F`。头文件里虽然定义了这些类型的枚举，但库实现不接受——判断能力必须实际调用，不能只看头文件。

**多卡之间怎么互联？**

曦云系列支持 MetaXLink 全互连。注意 `cudaMemcpyPeer` 实测只有约 55 GB/s，疑似走 PCIe 而非 MetaXLink；验证真实链路带宽需用 MCCL 的 all-to-all 测试。

## 贡献

欢迎补充项目、修正描述、更新实测数据。提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

收录标准：

- 与沐曦 GPU 直接相关（适配、工具、文档、基准）
- 有公开代码或可访问的文档
- 描述准确，不夸大能力
- 已归档或无实质内容的项目不予收录

## 说明

本清单为社区整理，与沐曦集成电路（上海）股份有限公司无隶属关系。
条目描述来自各项目公开信息，实测数据均标注来源与测量方法。
商标与项目版权归各自所有者。
