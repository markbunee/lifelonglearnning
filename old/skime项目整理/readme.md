这个项目代码支持的四种 MCP 协议及其特点如下：

## 1. **Hosted MCP（托管型 MCP）**

- **特点**：将完整的工具调用往返过程推送到 OpenAI 的基础设施中执行
- **适用场景**：需要让 OpenAI 的 Responses API 代表模型调用可公开访问的外部 MCP 服务器
- **工作方式**：通过 `HostedMCPTool` 将服务器标签（和可选的连接器元数据）转发给 Responses API，模型自行列出并调用远程服务器的工具，无需额外的 Python 进程回调
- **额外功能**：支持可选的审批流程（`require_approval`），支持 OpenAI Connectors 连接器

## 2. **Streamable HTTP MCP**

- **特点**：基于 Streamable HTTP 协议的传输方式
- **适用场景**：自己管理网络连接，或者在自有基础设施中运行服务器以保持低延迟
- **工作方式**：通过 `MCPServerStreamableHttp` 建立连接，可配置超时、自动重试、工具过滤等
- **优势**：适合本地或远程部署，提供完整的连接控制

## 3. **HTTP with SSE MCP**

- **特点**：基于 HTTP + Server-Sent Events（服务器推送事件）的传输方式
- **适用场景**：MCP 服务器实现了 HTTP with SSE 传输协议
- **工作方式**：通过 `MCPServerSse` 进行通信，API 与 Streamable HTTP 服务器基本相同
- **配置项**：支持 URL、自定义请求头、工具列表缓存等

## 4. **stdio MCP**

- **特点**：基于标准输入/输出（stdin/stdout）的本地进程间通信
- **适用场景**：MCP 服务器作为本地子进程运行，或者服务器只暴露命令行入口点
- **工作方式**：通过 `MCPServerStdio` 启动本地进程，SDK 负责维护管道并在上下文管理器退出时自动关闭
- **优势**：适合快速原型验证和本地文件系统类工具

------

**总结对比**：

| 协议类型        | 执行位置    | 通信方式      | 适用场景                     |
| :-------------- | :---------- | :------------ | :--------------------------- |
| Hosted          | OpenAI 云端 | Responses API | 外部公开服务器，无需本地处理 |
| Streamable HTTP | 本地/远程   | HTTP 流式     | 自建基础设施，低延迟需求     |
| SSE             | 本地/远程   | HTTP + SSE    | 兼容 SSE 的服务器            |
| stdio           | 本地        | 标准输入输出  | 本地命令行工具、快速验证     |



openai agents SDK

它是 OpenAI 推出的一个「Agent 框架」，核心目标是：

- 管理多 Agent
- Tool Calling
- MCP（Model Context Protocol）
- Memory / Context
- 工作流编排
- Streaming
- 多模型路由
- Agent handoff（代理协作）

调 GitHub

调数据库

调浏览器

调本地文件

调搜索引擎

调医院 HIS

调企业 ERP

调内部 API





现在主流 Agent 框架基本可以分成几类：

- 官方模型厂商系
- Workflow / 多 Agent 编排系
- 图执行（Graph）系
- Auto-Agent 自治体系
- 企业生产系

目前 2026 年比较主流的是下面这些。

------

# 一、OpenAI 系

## 1. [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/?utm_source=chatgpt.com)

现在非常热门。

特点：

- OpenAI 官方
- 原生支持 Responses API
- MCP 支持最强
- Tool Calling 很轻量
- 多 Agent handoff
- tracing
- structured output

适合：

- 企业 Agent
- MCP 工具系统
- 医疗 AI
- AI 中台
- 多工具 orchestrator

优点：

- 简洁
- 官方维护
- 未来兼容性最好
- 非常适合生产

缺点：

- 生态还没 LangChain 大
- workflow 不如 LangGraph 强

现在很多人已经：

```
LangChain → Agents SDK
```

迁移。

------

# 二、LangChain 系

## 2. [LangChain](https://www.langchain.com/?utm_source=chatgpt.com)

最老牌。

Agent 爆发的起点之一。

特点：

- 工具链极其丰富
- integrations 超多
- 文档生态大
- 社区最大

适合：

- 快速原型
- RAG
- 各种三方接入

问题：

- abstraction 太厚
- 历史包袱重
- 很容易“套娃”
- debug 痛苦

现在大家更多：

- 用它的生态
- 不一定用它的 Agent

------

## 3. [LangGraph](https://www.langchain.com/langgraph?utm_source=chatgpt.com)

现在 LangChain 真正的核心。

非常重要。

它是：

```
状态机 + DAG + Agent
```

核心思想：

```
节点 = Agent / Tool
边 = 条件流转
```

例如：

```
Planner
   ↓
Researcher
   ↓
Coder
   ↓
Reviewer
```

适合：

- 长流程 AI
- 多阶段推理
- 医疗流程
- 企业审批流
- 可恢复 Agent

这是目前：

> 企业级 Agent 最主流框架之一

很多大厂内部都在类似做。

优点：

- 稳定
- 可控
- 状态管理强
- 人类可介入

缺点：

- 学习成本高
- 写起来偏工程化

------

# 三、多 Agent 框架

## 4. [CrewAI](https://www.crewai.com/?utm_source=chatgpt.com)

最近两年很火。

核心：

```
Role Playing Agent
```

例如：

```
PM Agent
Research Agent
Engineer Agent
Reviewer Agent
```

自动协作。

非常适合 Demo。

优点：

- 上手快
- 很直观
- 多 Agent 很自然

缺点：

- 大规模生产稳定性一般
- workflow 不够严谨
- 长链容易失控

适合：

- AI 团队模拟
- 自动内容生产
- 多角色系统

------

## 5. [Microsoft AutoGen](https://microsoft.github.io/autogen/?utm_source=chatgpt.com)

微软出的。

核心：

```
Agent conversation
```

Agent 之间像聊天一样协作。

例如：

```
Assistant
↔
Coder
↔
Executor
```

非常适合：

- 自动 coding
- research agent
- 自动调试

优点：

- 多 Agent 非常强
- code execution 很成熟

缺点：

- 控制复杂
- token 消耗大

------

# 四、Google 系

## 6. [Google ADK (Agent Development Kit)](https://google.github.io/adk-docs/?utm_source=chatgpt.com)

Google 新推的。

主要配 Gemini。

特点：

- 原生 Gemini
- Workflow
- Tool
- Session
- Memory

但生态目前还不如 OpenAI / LangGraph。

------

# 五、企业工作流系

## 7. [LlamaIndex](https://www.llamaindex.ai/?utm_source=chatgpt.com)

原来做 RAG。

现在越来越 Agent 化。

特点：

- 知识库特别强
- retrieval pipeline 强
- 文档 Agent 很成熟

适合：

- 企业知识库
- 医疗文献
- 数据分析

------

## 8. [Haystack](https://haystack.deepset.ai/?utm_source=chatgpt.com)

德国 deepset 的。

偏企业搜索。

现在也支持 Agent。

特点：

- Pipeline 强
- 企业检索成熟

------

# 六、国产常见 Agent 框架

------

## 9. [Dify](https://dify.ai/?utm_source=chatgpt.com)

现在国内最火。

特点：

- 可视化
- 工作流
- RAG
- API 发布
- SaaS 化

很多公司：

```
不会写代码
↓
直接 Dify
```

适合：

- AI 应用平台
- 企业内部知识库
- 工作流

缺点：

- 深度定制有限
- 真复杂 Agent 不够强

------

## 10. [FastGPT](https://fastgpt.io/?utm_source=chatgpt.com)

国内也很多用。

偏：

```
知识库 + 工作流
```

| 场景          | 推荐                   |
| ------------- | ---------------------- |
| 快速 Demo     | CrewAI                 |
| 企业 Agent    | OpenAI Agents SDK      |
| 长工作流      | LangGraph              |
| 多 Agent 协作 | AutoGen                |
| RAG + 文档    | LlamaIndex             |
| 低代码平台    | Dify                   |
| MCP 工具生态  | OpenAI Agents SDK      |
| 医疗 AI       | LangGraph + Agents SDK |
| AI Coding     | AutoGen                |
| 企业知识库    | Dify / LlamaIndex      |

- **要可控、可观测、可上线**：首推 **LangGraph**（编排/持久化/调试/人-在-环），可与 **PydanticAI**（强类型与校验）组合；如果你在微软生态，**Semantic Kernel Agent Framework** 也很稳。
- **要多智能体“团队协作/对话式协同”**：看 **AutoGen**（微软系、多智能体会话、生态工具齐全）、**CrewAI**（“crew/role/task”范式，上手快，社区活跃）、**Langroid**（学术派、Actor 风格消息传递）。
- **要做“角色扮演/仿真/社会学或行为研究”**：**CAMEL**（角色扮演与“缩放定律”社区）与 **AgentVerse/MetaGPT**（多角色 SOP/仿真）更合适。
- **要极简原型/教学或小巧可嵌**：**smolagents**（Hugging Face，千行左右代码，简单直接）、**OpenAI Swarm**（教育性质的多 Agent 交接范式），若需持久化可看 **DurableSwarm**。
- **老牌“自主代理”与平台**：**AutoGPT / SuperAGI / AGiXT** 提供持续运行与插件/工具生态，但生产稳定性与工程化支持相对弱于上面几家编排框架。
- **RAG/检索 + Agent 统一**：做文档/知识工作流优先时，**LlamaIndex Agents** 或 **Haystack Agents** 更顺手。
- **官方 API 侧的“Agent 能力”**：OpenAI **Responses API + Agents SDK** 新一代官方路径（逐步替代 Assistants API），适合作为底座/工具层与上面编排框架组合。

## 核心理念与技术风格（通俗版）

### 1) “**编排/状态机**”派：把 Agent 当**流程**来建

- **LangGraph**：用**有向图（Graph）**建可恢复、可回放、可插人类反馈的长流程；强调**可控性与可观测性**。适合你把 Agent 视为**业务编排**而非“神秘黑箱”。
- **Semantic Kernel Agent Framework**：在 SK 生态里提供 Agent/Planner 能力，承接 .NET/JS/Python 企业开发的工程化需求。
- **PydanticAI**：把 **“强类型+验证/Guardrails”** 的 Pydantic 思想带入 Agent，输入输出都“可验可断言”，工程师体验像写 FastAPI。可与任何编排（如 LangGraph）搭配。

### 2) “**多智能体对话/团队**”派：把 Agent 当**角色**来搭

- **AutoGen**：一切围绕**多智能体对话**（Agent 彼此聊），从对话模式就能表达协作/评审/循环改进；配套 **AutoGen Studio/Bench**。
- **CrewAI**：把系统抽象为 **“Crew（团队）-Role（角色）-Task（任务）”**，YAML/CLI 体验友好，上手极快。AWS 也写了选型指南。
- **Langroid**：学术派，多 Agent **消息传递**范式（Actor 思想），代码简洁可控。
- **CAMEL**：**角色扮演（role-playing）**与“社会化”研究起家，适合仿真/数据合成/行为研究。

### 3) “**极简/实验**”派：把 Agent 当**可插积木**

- **smolagents**：**千来行**的极简库，强调“**写代码即推理**”（actions in code），很好嵌入现有工程。
- **OpenAI Swarm / DurableSwarm**：面向**模式探索/教育**的多 Agent 交接与“轻编排”，Durable 版补上**持久/重试**。

### 4) “**RAG+Agent 合一**”

- **LlamaIndex Agents** 与 **Haystack Agents** 均把**检索/索引/评估**与 Agent 循环打通，适合保险理赔、知识抽取、问答流等文档流程。 (

### 5) “**持续自主**”与平台

- **AutoGPT / SuperAGI / AGiXT**：强调“**持续运行**”与插件/工具生态，擅长自动化脚本、看板与可视化，但流程**确定性与调试体验**相对不足。



- **银行智能风控 Agent**：
  Python 处理 NLP 风险分析 → Java 事务层冻结账户/发送警报
- **医疗诊断 Agent**：
  Python 调用医学大模型 → Java 对接 HIS 系统（符合 HIPAA 审计）
- **工业 IoT Agent**：
  Python 分析传感器时序数据 → Java 控制 PLC 设备（硬实时要求）