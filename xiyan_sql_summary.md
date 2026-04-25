# XiYan-SQL 复现成果总结

## 项目概述

基于论文 [XiYan-SQL: Simple and Effective Text-to-SQL with Schema Filter and Multi-Generator Ensemble (2025)](https://arxiv.org/html/2507.04701v2) 的实现。

---

## 已实现模块

### 1. Schema Filter 模块（Algorithm 1）

**文件**: `schema_filter.py`

**实现内容**:
- LLM 关键词提取（使用大模型从问题和证据中提取 5-10 个关键词）
- 嵌入相似度匹配（使用 text-embedding-v4 模型计算余弦相似度）
- 迭代列选择算法（生成多个 schema 版本）
- PK/FK 列识别（自动识别主键和外键列）
- 多路径检索（表、列、值的相似度匹配）

**与论文的一致性**: ✅ 完全一致

---

### 2. Multi-Generator Ensemble（Algorithm 2）

**文件**: `generate_sql.py`

**实现内容**:
- 5 个不同风格的生成器：
  - **standard**: 标准风格
  - **detailed**: 详细思考风格（类似 CoT）
  - **simple**: 简洁风格
  - **structural**: 结构变体风格（使用 CTE 等复杂结构）
  - **icl**: ICL 风格（带 few-shot 示例）← 新增
- 多 schema 版本生成（支持 p_s 个版本）
- Self-refine 机制（生成失败时自动重试 1 次）

**与论文的差异**:
- ⚠️ 论文使用 4 个微调模型 + 1 个 ICL-based 模型（如 GPT-4o）
- ⚠️ 当前使用同一个模型（qwen3.6-plus）的不同提示词风格
- ⚠️ 缺少真正的微调模型（需要 SFT 训练）

---

### 3. Candidate Reorganization Strategy（Algorithm 3）

**文件**: `generate_sql.py`

**实现内容**:
- SQL 标准化（de-formalize，移除表面风格差异）
- 按执行结果聚类（使用标准化 SQL 作为近似）
- 组间排序（按组大小降序）
- 组内排序（按生成器性能排序）
- 主导组检查（最大组 >= 总候选数的一半）
- 重组策略（有主导组时使用所有候选，否则每组选最短 SQL）

**与论文的一致性**: ✅ 基本一致（缺少真实执行结果，使用 SQL 本身近似）

---

### 4. Selection Model（Algorithm 3）

**文件**: `generate_sql.py`

**实现内容**:
- 使用 LLM 作为选择模型
- 输入：问题 + schema union + evidence + 重组后的候选列表
- 输出：选择的最佳 SQL

**与论文的一致性**: ✅ 一致

---

## 主要差异总结

| 模块 | 论文实现 | 当前实现 | 差异程度 |
|------|---------|---------|---------|
| Schema Filter | LLM 关键词 + 嵌入相似度 + 迭代选择 | 相同 | ✅ 无差异 |
| 生成器类型 | 4 个微调模型 + 1 个 ICL 模型 | 5 个提示词风格（同一模型） | ⚠️ 较大 |
| ICL 示例 | 使用真实 few-shot 示例 | 已添加 3 个 dev.json 示例 | ✅ 已补充 |
| 执行结果聚类 | 真实数据库执行结果 | 使用标准化 SQL 近似 | ⚠️ 中等 |
| 候选重组 | 完整实现 | 完整实现 | ✅ 无差异 |
| 选择模型 | LLM 选择 | LLM 选择 | ✅ 无差异 |

---

## 核心架构

```
问题 + 证据 + 完整 Schema
         ↓
   Schema Filter (Algorithm 1)
   → 生成 p_s 个 schema 版本
         ↓
   Multi-Generator Ensemble (Algorithm 2)
   → p_s × p_m = 10 个候选 SQL
         ↓
   Candidate Reorganization (Algorithm 3)
   → De-formalize → 聚类 → 排序 → 重组
         ↓
   Selection Model (Algorithm 3)
   → LLM 选择最佳 SQL
         ↓
      最终输出
```

---

## 项目文件

- `schema_filter.py`: Schema Filter 模块
- `generate_sql.py`: 主生成器（包含 Multi-Generator Ensemble、Candidate Reorganization、Selection Model）
- `batch_generate.py`: 批量生成工具
- `bird_dataset.py`: BIRD 数据集加载工具

---

## 下一步建议

1. **微调模型**: 使用多任务联合训练策略训练 4 个不同格式的 SQL 生成模型
2. **真实执行**: 接入真实数据库执行结果用于聚类
3. **ICL 示例优化**: 使用检索式动态选择 few-shot 示例
4. **性能评估**: 在 BIRD benchmark 上测试准确率
