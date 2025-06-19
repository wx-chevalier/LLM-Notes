> [2025~Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

# Building effective agents

#### 一、核心结论

Anthropic 通过与跨行业团队合作发现，构建有效 LLM 智能体的关键在于使用**简单可组合的模式**，而非复杂框架。成功的实现通常从基础 LLM API 开始，仅在必要时增加复杂度，重点关注增强型 LLM（含检索、工具、记忆）的应用及工作流设计。

#### 二、智能体与工作流的定义

| **类型** | **定义**                                         | **核心区别**             |
| -------- | ------------------------------------------------ | ------------------------ |
| 工作流   | 通过预定义代码路径编排 LLM 和工具                | 流程固定，适用于明确任务 |
| 智能体   | LLM 动态指导自身流程和工具使用，控制任务完成方式 | 动态自主，适用于灵活场景 |

#### 三、构建模块与工作流模式

1.  **增强型 LLM**

- 基础模块：LLM 集成**检索、工具、记忆**功能，可主动生成查询、选择工具、决定信息保留

- 实现建议：通过 Model Context Protocol 集成第三方工具，定制化能力并提供清晰接口

![The augmented LLM](https://ngte-superbed.oss-cn-beijing.aliyuncs.com/uPic/AdDWqVPeN8pl.png)

2.  **提示链（Prompt Chaining）**

- 流程：将任务分解为连续步骤，每步 LLM 处理前一步输出，可添加程序检查点

- 适用场景：任务可清晰分解为固定子任务，如生成营销文案并翻译、撰写文档大纲后成文

![The prompt chaining workflow](https://ngte-superbed.oss-cn-beijing.aliyuncs.com/uPic/6ZRwctR3ekD6.png)

3.  **路由（Routing）**

- 流程：分类输入并导向专门后续任务，分离不同类型处理逻辑

- 适用场景：客户服务查询分类（常规问题、退款请求等）、按问题难度路由至不同模型

![The routing workflow](https://ngte-superbed.oss-cn-beijing.aliyuncs.com/uPic/qexzivyRBoIv.png)

4.  **并行化（Parallelization）**

- **分块（Sectioning）**：将任务拆分为独立子任务并行处理，如内容审核与核心响应分离

- **投票（Voting）**：多次运行同一任务聚合结果，如代码漏洞审查、内容合规性评估

![The parallelization workflow](https://ngte-superbed.oss-cn-beijing.aliyuncs.com/uPic/WQJKbPo8dCbK.png)

5.  **协调器 - 工作器（Orchestrator-Workers）**

- 流程：中央 LLM 动态分解任务并委派给工作 LLM，合成结果

- 适用场景：复杂编码任务（需修改多文件）、多源信息搜索与分析

![The orchestrator-workers workflow](https://ngte-superbed.oss-cn-beijing.aliyuncs.com/uPic/MBKifCCQ1s3t.png)

6.  **评估器 - 优化器（Evaluator-Optimizer）**

- 流程：LLM 生成响应并由另一 LLM 迭代评估优化，类似人类写作修订

- 适用场景：文学翻译润色、多轮搜索任务完善信息

![The evaluator-optimizer workflow](https://ngte-superbed.oss-cn-beijing.aliyuncs.com/uPic/JUXq4d3MsEWj.png)

#### 四、智能体的应用与实施原则

1.  **适用场景**

- 开放式问题（步骤不可预测），如编码解决 SWE-bench 任务、计算机使用自动化

- 需动态规划和工具使用，依赖环境反馈循环

2.  **实施原则**

- **保持简单**：从基础 prompt 开始，仅在必要时添加多步流程

- **透明性**：显式展示智能体规划步骤

- **接口设计**：精心构建智能体 - 计算机接口（ACI），如工具文档、格式优化

#### 五、附录：实践案例与工具提示工程

1.  **客户支持**

- 结合聊天界面与工具集成（拉取客户数据、处理退款），按成功解决率收费

2.  **编码智能体**

- 解决 GitHub 问题，通过自动化测试验证功能，但需人类审查确保系统一致性

3.  **工具提示工程要点**

- 选择模型熟悉的格式（如 Markdown 而非 JSON 代码）

- 提供示例、边缘案例和清晰参数说明

- 测试工具使用并优化，如强制使用绝对文件路径避免错误

![High-level flow of a coding agent](https://ngte-superbed.oss-cn-beijing.aliyuncs.com/uPic/vQf6S0DbjyJc.png)

### 关键问题与答案

1.  **问题：工作流与智能体的核心区别是什么？**

    答案：工作流是通过预定义代码路径编排 LLM 和工具，适用于流程固定的明确任务；而智能体由 LLM 动态指导自身流程和工具使用，自主控制任务完成方式，适用于需要灵活性和模型驱动决策的场景。

2.  **问题：构建有效智能体的三个核心原则是什么？**

    答案：① 保持智能体设计简单，从基础 prompt 开始逐步迭代；② 优先透明性，显式展示智能体的规划步骤；③ 精心设计智能体 - 计算机接口（ACI），通过工具文档和测试优化交互。

3.  **问题：在工具提示工程中，如何优化 LLM 对工具的使用？**

    答案：① 选择模型熟悉的格式（如接近互联网文本的 Markdown 而非复杂 JSON）；② 提供详细示例、边缘案例和参数说明，类似为初级开发者编写文档；③ 测试工具使用并迭代优化，例如将相对文件路径改为绝对路径以减少错误。

以上总结涵盖了网页中构建有效智能体的主要内容。你可以告诉我是否需要对某个部分进一步细化，或者有其他修改需求。
