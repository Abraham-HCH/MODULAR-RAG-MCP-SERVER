### 下一个任务：C4 - Splitter 集成（调用 Libs）

**任务描述**：
- 将 Splitter 的抽象接口与实现集成到 Ingestion Pipeline 中。
- 确保调用 Libs 层的 SplitterFactory，支持多种 Splitter 实现。

**验收标准**：
1. 编写单元测试，验证 Splitter 集成的正确性。
2. 确保 Splitter 能够正确处理样例文档并生成 Chunk。

**测试方法**：
- 使用 pytest 编写单元测试，覆盖 Splitter 的主要功能。
- 提供样例文档，验证生成的 Chunk 是否符合预期。

**预计完成时间**：1 小时。