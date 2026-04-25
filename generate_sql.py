#!/usr/bin/env python3
"""
XiYan-SQL: 完整实现

基于论文: XiYan-SQL: Simple and Effective Text-to-SQL with Schema Filter and Multi-Generator Ensemble (2025)

架构:
1. Schema Filter 模块 (已在 schema_filter.py 中实现)
2. Multi-Generator Ensemble (多生成器集成) - 算法2
3. Candidate Reorganization Strategy (候选重组策略) - 算法3
4. Selection Model (选择模型) - 算法3
"""

import os
import re
import json
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from openai import OpenAI

from bird_dataset import BirdDataset
from schema_filter import SchemaFilter, TableSchema, FilteredSchemaSet


# ==============================
# 数据结构定义
# ==============================

@dataclass
class SQLCandidate:
    """SQL 候选对象"""
    sql: str
    source: str  # 来源描述（如 "schema0_gen1"）
    schema_version: Optional[int] = None
    generator_id: Optional[int] = None  # 生成器ID（用于排序）
    execution_result: Optional[Any] = None  # 执行结果（用于聚类）
    has_error: bool = False  # 是否有执行错误
    error_message: Optional[str] = None  # 错误信息
    is_valid: bool = True  # SQL 是否有效


# ==============================
# 候选重组器 (Candidate Reorganizer) - 算法3
# ==============================

class CandidateReorganizer:
    """
    候选重组器（论文: Candidate Reorganization Strategy - 算法3）
    
    职责:
    1. 对候选 SQL 进行 de-formalize（标准化格式）
    2. 按执行结果聚类
    3. 组间排序（按组大小降序）
    4. 组内排序（按生成器性能排序）
    5. 根据是否有主导组决定重组策略
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-bb901ef8d7e44cb0be1c535e137974c4")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def _deformalize_sql(self, sql: str) -> str:
        """
        De-formalize: 标准化 SQL 格式，移除表面的风格差异
        （论文: de-formalization process）
        """
        if not sql:
            return ""
        # 移除多余空格，统一大小写
        normalized = re.sub(r'\s+', ' ', sql.strip()).upper()
        return normalized
    
    def _group_by_execution_result(self, candidates: List[SQLCandidate]) -> List[List[SQLCandidate]]:
        """
        按执行结果聚类（论文: groupby(L, R)）
        
        Returns:
            聚类后的候选列表 [[c1, c2], [c3], ...]
        """
        # 按执行结果分组
        result_to_candidates = defaultdict(list)
        
        for candidate in candidates:
            # 使用执行结果作为键（如果没有执行结果，使用 SQL 本身）
            if candidate.execution_result is not None:
                key = str(candidate.execution_result)
            else:
                # 如果没有执行，使用标准化后的 SQL 作为近似
                key = self._deformalize_sql(candidate.sql)
            
            result_to_candidates[key].append(candidate)
        
        # 转换为列表
        clusters = list(result_to_candidates.values())
        
        return clusters
    
    def _sort_clusters_by_size(self, clusters: List[List[SQLCandidate]]) -> List[List[SQLCandidate]]:
        """
        组间排序：按组大小降序（论文: sort(C) by size, descending）
        """
        return sorted(clusters, key=lambda x: -len(x))
    
    def _sort_within_cluster(self, cluster: List[SQLCandidate], generator_order: List[int]) -> List[SQLCandidate]:
        """
        组内排序：按生成器性能排序（论文: sort(C_i') order by O）
        
        Args:
            cluster: 候选列表
            generator_order: 生成器性能排序（性能高的在前）
        """
        # 为每个生成器分配优先级
        generator_priority = {gen_id: idx for idx, gen_id in enumerate(generator_order)}
        
        def sort_key(candidate):
            if candidate.generator_id is not None:
                return generator_priority.get(candidate.generator_id, 999)
            return 999
        
        return sorted(cluster, key=sort_key)
    
    def reorganize(
        self, 
        candidates: List[SQLCandidate], 
        generator_order: Optional[List[int]] = None
    ) -> Tuple[List[SQLCandidate], bool]:
        """
        执行候选重组策略（论文: Algorithm 3）
        
        Args:
            candidates: 候选 SQL 列表
            generator_order: 生成器性能排序（可选，默认按 ID 排序）
            
        Returns:
            (重组后的候选列表, 是否有主导组)
        """
        if not candidates:
            return [], False
        
        print(f"\n{'='*60}")
        print("Candidate Reorganization Strategy (Algorithm 3)")
        print(f"{'='*60}")
        
        # 1. De-formalize 所有候选
        print("\nStep 1: De-formalize candidates")
        for candidate in candidates:
            original = candidate.sql
            candidate.sql = self._deformalize_sql(candidate.sql)
            if original != candidate.sql:
                print(f"  [{candidate.source}] De-formalized")
        
        # 2. 按执行结果聚类
        print("\nStep 2: Group by execution result")
        clusters = self._group_by_execution_result(candidates)
        print(f"  Found {len(clusters)} clusters")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {len(cluster)} candidates")
        
        # 3. 如果只有一个组，选最短的 SQL
        if len(clusters) == 1:
            print("\nStep 3: Only one cluster, selecting shortest SQL")
            cluster = clusters[0]
            cluster.sort(key=lambda x: len(x.sql))
            best = cluster[0]
            print(f"  Selected: {best.source} (length: {len(best.sql)})")
            return [best], False
        
        # 4. 组间排序（按组大小降序）
        print("\nStep 4: Inter-group sorting (by size)")
        sorted_clusters = self._sort_clusters_by_size(clusters)
        
        # 5. 组内排序（按生成器性能）
        print("\nStep 5: Intra-group sorting (by generator performance)")
        if generator_order is None:
            # 默认按 generator_id 排序
            generator_order = sorted(set(
                c.generator_id for c in candidates if c.generator_id is not None
            ))
        
        final_clusters = []
        for i, cluster in enumerate(sorted_clusters):
            sorted_cluster = self._sort_within_cluster(cluster, generator_order)
            final_clusters.append(sorted_cluster)
            print(f"  Cluster {i+1} ({len(cluster)} candidates) sorted")
        
        # 6. 检查是否有主导组
        largest_cluster_size = len(final_clusters[0])
        total_candidates = len(candidates)
        has_dominant_group = largest_cluster_size >= (total_candidates + 1) // 2
        
        print(f"\nStep 6: Check for dominant group")
        print(f"  Largest cluster: {largest_cluster_size}/{total_candidates}")
        print(f"  Has dominant group: {has_dominant_group}")
        
        # 7. 重组候选列表
        print("\nStep 7: Reorganize candidates")
        reorganized_candidates = []
        
        if has_dominant_group:
            # 如果有主导组，按重组顺序添加所有候选
            print("  Strategy: Add all candidates in order")
            for cluster in final_clusters:
                reorganized_candidates.extend(cluster)
        else:
            # 否则，每个组选最短的 SQL
            print("  Strategy: Select shortest from each cluster")
            for cluster in final_clusters:
                cluster.sort(key=lambda x: len(x.sql))
                reorganized_candidates.append(cluster[0])
        
        print(f"\nReorganization complete. {len(reorganized_candidates)} candidates.")
        
        return reorganized_candidates, has_dominant_group


# ==============================
# 选择模型 (Selection Model) - 算法3
# ==============================

class SelectionModel:
    """
    选择模型（论文: Selection Model - 算法3）
    
    职责:
    1. 接收重组后的候选列表
    2. 结合问题、schema union、evidence 进行选择
    3. 输出最佳 SQL
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, model: str = "qwen3.6-plus"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-bb901ef8d7e44cb0be1c535e137974c4")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def _build_selection_prompt(
        self,
        question: str,
        schema_union_str: str,
        evidence: str,
        candidates: List[SQLCandidate]
    ) -> str:
        """
        构建选择模型的提示词
        
        Returns:
            提示词字符串
        """
        # 构建候选列表
        candidates_str = ""
        for i, candidate in enumerate(candidates):
            candidates_str += f"Candidate {i+1} [{candidate.source}]:\n```sql\n{candidate.sql}\n```\n\n"
        
        prompt = f"""You are an expert SQL selector. Given a question, database schema, evidence, and multiple SQL candidates, select the best SQL query that correctly answers the question.

【Question】
{question}

【Database Schema】
{schema_union_str}

【Evidence】
{evidence if evidence else '(No additional evidence)'}

【SQL Candidates】
{candidates_str}

Please analyze each candidate carefully and select the best one. Consider:
1. Does the SQL correctly answer the question?
2. Is the SQL syntactically correct?
3. Does the SQL use the correct tables and columns?
4. Is the SQL efficient and well-structured?

Output ONLY the number of the best candidate (e.g., "1", "2", "3", etc.).
""".strip()
        
        return prompt
    
    def _extract_selection(self, response: str, num_candidates: int) -> int:
        """
        从模型响应中提取选择的候选编号
        
        Returns:
            候选编号（从1开始）
        """
        # 尝试提取数字
        match = re.search(r'\b(\d+)\b', response.strip())
        if match:
            num = int(match.group(1))
            if 1 <= num <= num_candidates:
                return num
        
        # 如果提取失败，默认选第一个
        return 1
    
    def select_best(
        self,
        question: str,
        schema_union_str: str,
        evidence: str,
        candidates: List[SQLCandidate]
    ) -> Optional[SQLCandidate]:
        """
        选择最佳 SQL（论文: Algorithm 3 - Selection Model）
        
        Args:
            question: 用户问题
            schema_union_str: schema union 的字符串表示
            evidence: 额外证据
            candidates: 重组后的候选列表
            
        Returns:
            最佳候选
        """
        if not candidates:
            return None
        
        if len(candidates) == 1:
            print(f"\n{'='*60}")
            print("Selection Model: Only one candidate, selecting it")
            print(f"{'='*60}")
            return candidates[0]
        
        print(f"\n{'='*60}")
        print("Selection Model: Choosing best SQL")
        print(f"{'='*60}")
        
        # 构建提示词
        prompt = self._build_selection_prompt(
            question, schema_union_str, evidence, candidates
        )
        
        print(f"\nSending {len(candidates)} candidates to selection model...")
        
        try:
            # 调用选择模型
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            response_text = completion.choices[0].message.content
            print(f"Selection model response: {response_text[:200]}...")
            
            # 提取选择
            selected_idx = self._extract_selection(response_text, len(candidates))
            best_candidate = candidates[selected_idx - 1]
            
            print(f"\nSelected candidate {selected_idx}: {best_candidate.source}")
            print(f"SQL: {best_candidate.sql[:100]}...")
            
            return best_candidate
            
        except Exception as e:
            print(f"⚠️ Selection model failed: {e}")
            print("Falling back to first candidate")
            return candidates[0]


# ==============================
# 多生成器集成 (Multi-Generator Ensemble) - 算法2
# ==============================

class XiYanSQLGenerator:
    """
    XiYan-SQL 主生成器
    
    包含完整架构:
    1. Schema Filter (过滤 schema)
    2. Multi-Generator Ensemble (多生成器集成) - 算法2
    3. Candidate Reorganization (候选重组) - 算法3
    4. Selection Model (选择模型) - 算法3
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen3.6-plus",
        selection_model: str = "qwen3.6-plus"
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-bb901ef8d7e44cb0be1c535e137974c4")
        self.base_url = base_url
        self.model = model
        
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        
        # 初始化各个组件
        self.schema_filter = SchemaFilter(api_key=self.api_key, base_url=self.base_url)
        self.reorganizer = CandidateReorganizer(api_key=self.api_key, base_url=self.base_url)
        self.selection_model = SelectionModel(
            api_key=self.api_key, base_url=self.base_url, model=selection_model
        )
    
    def _build_prompt_standard(self, question: str, schema_str: str, evidence: str = "", dialect: str = "sqlite") -> str:
        """提示词: 标准风格"""
        return f"""You are now a {dialect} data analyst, and you are given a database schema as follows:

【Schema】
{schema_str}

【Question】
{question}

【Evidence】
{evidence if evidence else '(No additional evidence)'}

Please read and understand the database schema carefully, and generate an executable SQL based on the user's question and evidence. 
The generated SQL must be compatible with {dialect} dialect.
Please output the SQL ONLY, wrapped by ```sql and ```.
""".strip()
    
    def _build_prompt_detailed(self, question: str, schema_str: str, evidence: str = "", dialect: str = "sqlite") -> str:
        """提示词: 详细思考风格（类似 CoT）"""
        return f"""Task: Generate a {dialect} SQL query for the given question.

Database Schema:
{schema_str}

Question: {question}

Evidence: {evidence if evidence else '(No additional evidence)'}

Step 1: Analyze the question and understand what tables and columns are needed.
Step 2: Identify the relationships between tables (primary keys and foreign keys).
Step 3: Write the SQL query step by step.

Please generate the final SQL query ONLY, wrapped by ```sql and ```.
""".strip()
    
    def _build_prompt_simple(self, question: str, schema_str: str, evidence: str = "", dialect: str = "sqlite") -> str:
        """提示词: 简洁风格"""
        return f"""Generate {dialect} SQL for this question: {question}

Schema:
{schema_str}

Evidence: {evidence if evidence else 'None'}

Only output the SQL query wrapped by ```sql and ```.
""".strip()
    
    def _build_prompt_structural(self, question: str, schema_str: str, evidence: str = "", dialect: str = "sqlite") -> str:
        """提示词: 结构变体风格（使用 CTE 等复杂结构）"""
        return f"""Generate a {dialect} SQL query for the given question using advanced SQL structures.

Database Schema:
{schema_str}

Question: {question}

Evidence: {evidence if evidence else '(No additional evidence)'}

Requirements:
- Use Common Table Expressions (CTEs) with WITH clause when appropriate
- Use subqueries when they improve clarity
- Structure the query in a modular way

Please generate the final SQL query ONLY, wrapped by ```sql and ```.
""".strip()
    
    def _build_prompt_stylistic(self, question: str, schema_str: str, evidence: str = "", dialect: str = "sqlite") -> str:
        """提示词: 风格变体（特定的编码风格）"""
        return f"""Generate a {dialect} SQL query for the given question with specific coding style.

Database Schema:
{schema_str}

Question: {question}

Evidence: {evidence if evidence else '(No additional evidence)'}

Style Requirements:
- Use lowercase for SQL keywords (select, from, where, etc.)
- Use table aliases for all tables
- Format the query with proper indentation
- Use explicit JOIN syntax

Please generate the final SQL query ONLY, wrapped by ```sql and ```.
""".strip()
    
    def _extract_sql(self, response: str) -> str:
        """从模型响应中提取 SQL"""
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        sql_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
        if sql_match:
            return sql_match.group(1).strip()
        
        return response.strip()
    
    def _generate_single_sql(self, prompt: str, temperature: float = 0.1) -> Optional[str]:
        """单个 SQL 生成"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            response_text = completion.choices[0].message.content
            return self._extract_sql(response_text)
        except Exception as e:
            print(f"    ⚠️ 生成 SQL 失败: {e}")
            return None
    
    def _self_refine_sql(
        self,
        question: str,
        schema_str: str,
        evidence: str,
        dialect: str,
        previous_sql: str,
        error_message: str,
        build_prompt_fn
    ) -> Optional[str]:
        """
        Self-refine: 基于执行错误重新生成 SQL（算法2 第9-10行）
        """
        refine_prompt = f"""The following SQL query has an error. Please fix it.

【Schema】
{schema_str}

【Question】
{question}

【Evidence】
{evidence if evidence else '(No additional evidence)'}

【Previous SQL】
```sql
{previous_sql}
```

【Error】
{error_message}

Please generate a corrected SQL query ONLY, wrapped by ```sql and ```.
""".strip()
        
        return self._generate_single_sql(refine_prompt, temperature=0.1)
    
    def _generate_with_schema_versions(
        self,
        question: str,
        filtered_set: FilteredSchemaSet,
        evidence: str = "",
        dialect: str = "sqlite"
    ) -> List[SQLCandidate]:
        """
        使用多个 schema 版本 + 多个生成器 生成候选
        
        这是论文的 Multi-Generator Ensemble 核心部分（算法2）
        
        论文配置:
        - p_s = 2 (schema 版本数)
        - p_m = 5 (生成器数)
        - p_l = p_s * p_m = 10 (总候选数)
        """
        candidates = []
        
        # 定义多个生成器（对应论文的多格式 SQL 生成器）
        # 每个生成器有不同的提示词风格
        generators = [
            ("standard", self._build_prompt_standard, 0.1),      # 标准风格
            ("detailed", self._build_prompt_detailed, 0.2),      # 详细思考
            ("simple", self._build_prompt_simple, 0.1),          # 简洁风格
            ("structural", self._build_prompt_structural, 0.3),  # 结构变体（CTE等）
            ("stylistic", self._build_prompt_stylistic, 0.2),    # 风格变体
        ]
        
        # 算法2: Multiple SQL Generation
        # for i = 1 to p_s (schema 版本)
        for schema_idx, schema in enumerate(filtered_set.schemas):
            schema_str = self.schema_filter.schema_to_prompt(schema, include_descriptions=True)
            print(f"\n  Generating with schema version {schema_idx+1}/{len(filtered_set.schemas)}...")
            
            # for j = 1 to p_m (生成器)
            for gen_idx, (gen_name, build_fn, temp) in enumerate(generators):
                print(f"    Generator {gen_idx+1}/{len(generators)}: {gen_name}")
                
                # 第4行: Predict SQL l_ij ← f_Mj(Q, E, S_i)
                prompt = build_fn(question, schema_str, evidence, dialect)
                sql = self._generate_single_sql(prompt, temperature=temp)
                
                if sql:
                    # 第5-7行: 如果没有异常，添加到候选列表
                    source = f"schema{schema_idx}_gen{gen_idx}_{gen_name}"
                    candidate = SQLCandidate(
                        sql=sql,
                        source=source,
                        schema_version=schema_idx,
                        generator_id=gen_idx,
                        has_error=False
                    )
                    candidates.append(candidate)
                    print(f"      ✓ SQL generated successfully")
                else:
                    # 第8-11行: 如果有异常，重试1次（self-refine）
                    print(f"      ⚠️ SQL generation failed, attempting self-refine...")
                    
                    # 重试1次
                    retry_sql = self._self_refine_sql(
                        question, schema_str, evidence, dialect,
                        "", "Generation failed", build_fn
                    )
                    
                    if retry_sql:
                        source = f"schema{schema_idx}_gen{gen_idx}_{gen_name}_refined"
                        candidate = SQLCandidate(
                            sql=retry_sql,
                            source=source,
                            schema_version=schema_idx,
                            generator_id=gen_idx,
                            has_error=False
                        )
                        candidates.append(candidate)
                        print(f"      ✓ Self-refine successful")
                    else:
                        print(f"      ✗ Self-refine also failed")
        
        print(f"\n  Generated {len(candidates)} candidates from {len(filtered_set.schemas)} schemas × {len(generators)} generators")
        
        return candidates
    
    def _build_schema_union_str(self, filtered_set: FilteredSchemaSet) -> str:
        """
        构建 schema union 的字符串表示（论文: schema_union(S)）
        """
        if not filtered_set:
            return ""
        
        # 合并所有 schema 版本中的表和列
        all_tables = {}
        
        for schema in filtered_set.schemas:
            # schema 是 FilteredSchema 对象，包含 tables 列表和 relevant_columns 字典
            for table in schema.tables:
                table_name = table.name
                if table_name not in all_tables:
                    all_tables[table_name] = set()
                
                # 使用 relevant_columns 获取该表的相关列
                cols = schema.relevant_columns.get(table_name, table.columns)
                for col in cols:
                    all_tables[table_name].add(col)
        
        # 构建字符串
        parts = []
        for table_name, columns in all_tables.items():
            parts.append(f"Table: {table_name}")
            parts.append(f"Columns: {', '.join(sorted(columns))}")
            parts.append("")
        
        return "\n".join(parts)
    
    def generate_sql(
        self,
        question: str,
        full_schema: Dict[str, TableSchema],
        evidence: str = "",
        dialect: str = "sqlite",
        use_schema_filter: bool = True,
        use_multi_generator: bool = True,
        max_iterations: int = 1  # 对应论文的 p_s 参数（论文使用2）
    ) -> Dict[str, Any]:
        """
        主入口: 生成 SQL（完整 XiYan-SQL 流程）
        
        Args:
            question: 用户问题
            full_schema: 完整的数据库 schema
            evidence: 额外证据
            dialect: SQL 方言
            use_schema_filter: 是否使用 SchemaFilter
            use_multi_generator: 是否使用多生成器集成
            max_iterations: Schema Filter 迭代次数（即 schema 版本数量，论文使用2）
            
        Returns:
            结果字典
        """
        result = {
            "question": question,
            "evidence": evidence,
            "use_schema_filter": use_schema_filter,
            "use_multi_generator": use_multi_generator,
            "full_schema_tables": list(full_schema.keys()) if full_schema else [],
        }
        
        # ==============================
        # 第一步: Schema Filter (论文第一步)
        # ==============================
        if use_schema_filter and full_schema:
            print(f"\n{'='*60}")
            print("Step 1: Schema Filter")
            print(f"{'='*60}")
            
            filtered_set = self.schema_filter.filter_schema(
                question, full_schema, evidence=evidence, max_iterations=max_iterations
            )
            result["filtered_schema_set"] = filtered_set
        else:
            filtered_set = None
        
        # ==============================
        # 第二步: Multi-Generator Ensemble (论文第二步 - 算法2)
        # ==============================
        if use_multi_generator:
            print(f"\n{'='*60}")
            print("Step 2: Multi-Generator Ensemble (Algorithm 2)")
            print(f"{'='*60}")
            
            if filtered_set:
                candidates = self._generate_with_schema_versions(
                    question, filtered_set, evidence=evidence, dialect=dialect
                )
            else:
                candidates = []
                # 没有过滤的情况，也生成多个候选（简化处理）
                schema_parts = ["数据库 Schema:"]
                for table_name, table in full_schema.items():
                    schema_parts.append(f"\n表: {table_name}")
                    col_lines = []
                    for col in table.columns:
                        desc = table.get_column_description(col)
                        if desc:
                            clean_desc = re.sub(r'\s+', ' ', desc).strip()
                            col_lines.append(f"  - {col}: {clean_desc}")
                        else:
                            col_lines.append(f"  - {col}")
                    schema_parts.append("列:\n" + "\n".join(col_lines))
                    if table.primary_key:
                        schema_parts.append(f"主键: {table.primary_key}")
                    if table.foreign_keys:
                        for col, (ref_table, ref_col) in table.foreign_keys.items():
                            schema_parts.append(f"外键: {col} -> {ref_table}.{ref_col}")
                
                schema_str = "\n".join(schema_parts)
                
                # 使用多个生成器
                generators = [
                    ("standard", self._build_prompt_standard, 0.1),
                    ("detailed", self._build_prompt_detailed, 0.2),
                    ("simple", self._build_prompt_simple, 0.1),
                    ("structural", self._build_prompt_structural, 0.3),
                    ("stylistic", self._build_prompt_stylistic, 0.2),
                ]
                
                for gen_idx, (gen_name, build_fn, temp) in enumerate(generators):
                    prompt = build_fn(question, schema_str, evidence, dialect)
                    sql = self._generate_single_sql(prompt, temperature=temp)
                    if sql:
                        candidates.append(SQLCandidate(
                            sql=sql,
                            source=f"full_gen{gen_idx}_{gen_name}",
                            generator_id=gen_idx,
                            has_error=False
                        ))
            
            result["candidates"] = [
                {"sql": c.sql, "source": c.source} for c in candidates
            ]
            print(f"\nGenerated {len(candidates)} candidate SQLs")
            
            # ==============================
            # 第三步: Candidate Reorganization (论文第三步 - 算法3)
            # ==============================
            print(f"\n{'='*60}")
            print("Step 3: Candidate Reorganization (Algorithm 3)")
            print(f"{'='*60}")
            
            # 生成器性能排序（可以根据实际性能调整）
            generator_order = [0, 1, 2, 3, 4]  # 默认按 ID 排序
            
            reorganized_candidates, has_dominant_group = self.reorganizer.reorganize(
                candidates, generator_order=generator_order
            )
            
            result["reorganized_candidates"] = [
                {"sql": c.sql, "source": c.source} for c in reorganized_candidates
            ]
            result["has_dominant_group"] = has_dominant_group
            
            # ==============================
            # 第四步: Selection Model (论文第四步 - 算法3)
            # ==============================
            print(f"\n{'='*60}")
            print("Step 4: Selection Model (Algorithm 3)")
            print(f"{'='*60}")
            
            # 构建 schema union
            if filtered_set:
                schema_union_str = self._build_schema_union_str(filtered_set)
            else:
                schema_union_str = ""
                for table_name, table in full_schema.items():
                    schema_union_str += f"Table: {table_name}\n"
                    schema_union_str += f"Columns: {', '.join(table.columns)}\n\n"
            
            best_candidate = self.selection_model.select_best(
                question=question,
                schema_union_str=schema_union_str,
                evidence=evidence,
                candidates=reorganized_candidates
            )
            
            if best_candidate:
                result["sql"] = best_candidate.sql
                result["best_source"] = best_candidate.source
                print(f"\n✅ Final best SQL selected!")
            else:
                result["error"] = "No valid SQL candidates"
                print(f"\n❌ No valid SQL candidates")
        else:
            # 简化模式: 单生成器
            print(f"\n{'='*60}")
            print("Single Generator Mode (simplified)")
            print(f"{'='*60}")
            
            if use_schema_filter and filtered_set:
                best_schema = filtered_set.schemas[-1] if len(filtered_set.schemas) > 1 else filtered_set.best_schema
                schema_str = self.schema_filter.schema_to_prompt(best_schema, include_descriptions=True)
            else:
                schema_parts = ["数据库 Schema:"]
                for table_name, table in full_schema.items():
                    schema_parts.append(f"\n表: {table_name}")
                    col_lines = []
                    for col in table.columns:
                        desc = table.get_column_description(col)
                        if desc:
                            clean_desc = re.sub(r'\s+', ' ', desc).strip()
                            col_lines.append(f"  - {col}: {clean_desc}")
                        else:
                            col_lines.append(f"  - {col}")
                    schema_parts.append("列:\n" + "\n".join(col_lines))
                    if table.primary_key:
                        schema_parts.append(f"主键: {table.primary_key}")
                    if table.foreign_keys:
                        for col, (ref_table, ref_col) in table.foreign_keys.items():
                            schema_parts.append(f"外键: {col} -> {ref_table}.{ref_col}")
                schema_str = "\n".join(schema_parts)
            
            prompt = self._build_prompt_standard(question, schema_str, evidence, dialect)
            sql = self._generate_single_sql(prompt, temperature=0.1)
            result["sql"] = sql
            result["prompt"] = prompt
        
        return result


# ==============================
# 演示和测试
# ==============================

def demo_single_sample(
    generator: XiYanSQLGenerator,
    dataset: BirdDataset,
    sample_index: int,
    use_schema_filter: bool = True,
    use_multi_generator: bool = True,
    save_prompt: bool = True,
    prompt_output_dir: str = "./predictions/prompts"
):
    """演示单个样本"""
    sample = dataset.get_sample(sample_index)
    full_schema = dataset.get_schema(sample.db_id)
    
    print(f"\n{'='*100}")
    print(f"样本 #{sample.question_id} | 数据库: {sample.db_id} | 难度: {sample.difficulty}")
    print(f"{'='*100}")
    print(f"\n📝 问题: {sample.question}")
    if sample.evidence:
        print(f"\n📎 证据: {sample.evidence}")
    print(f"\n🎯 真实 SQL: {sample.sql}")
    
    # 生成 SQL
    result = generator.generate_sql(
        question=sample.question,
        full_schema=full_schema,
        evidence=sample.evidence,
        use_schema_filter=use_schema_filter,
        use_multi_generator=use_multi_generator
    )
    
    # 保存结果
    if save_prompt:
        if not os.path.exists(prompt_output_dir):
            os.makedirs(prompt_output_dir)
        
        filter_suffix = "_filtered" if use_schema_filter else "_full"
        multi_suffix = "_multi" if use_multi_generator else "_single"
        output_file = os.path.join(prompt_output_dir, f"sample_{sample.question_id}{filter_suffix}{multi_suffix}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=" * 80 + "\n")
            f.write(f"样本 #{sample.question_id}\n")
            f.write(f"数据库: {sample.db_id}\n")
            f.write(f"SchemaFilter: {use_schema_filter}\n")
            f.write(f"MultiGenerator: {use_multi_generator}\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"问题:\n{sample.question}\n\n")
            if sample.evidence:
                f.write(f"证据:\n{sample.evidence}\n\n")
            f.write(f"真实 SQL:\n{sample.sql}\n\n")
            
            if 'candidates' in result:
                f.write(f"=" * 80 + "\n")
                f.write(f"Generated Candidates ({len(result['candidates'])}):\n")
                f.write(f"=" * 80 + "\n\n")
                for i, cand in enumerate(result['candidates']):
                    f.write(f"Candidate {i+1} [{cand['source']}]:\n")
                    f.write(f"```sql\n{cand['sql']}\n```\n\n")
            
            if 'reorganized_candidates' in result:
                f.write(f"=" * 80 + "\n")
                f.write(f"Reorganized Candidates:\n")
                f.write(f"=" * 80 + "\n\n")
                for i, cand in enumerate(result['reorganized_candidates']):
                    f.write(f"Candidate {i+1} [{cand['source']}]\n")
                    f.write(f"```sql\n{cand['sql']}\n```\n\n")
            
            f.write(f"=" * 80 + "\n")
            f.write(f"Final Selected SQL:\n")
            f.write(f"=" * 80 + "\n\n")
            f.write(f"```sql\n{result.get('sql', '(None)')}\n```")
        
        print(f"\n💾 结果已保存到: {output_file}")
    
    # 显示结果
    print(f"\n{'='*100}")
    print("生成结果")
    print(f"{'='*100}")
    
    if result.get("error"):
        print(f"\n❌ 错误: {result['error']}")
    else:
        print(f"\n📄 SchemaFilter: {use_schema_filter}")
        print(f"🤖 Multi-Generator: {use_multi_generator}")
        
        if 'candidates' in result:
            print(f"\n📊 候选 SQL 数量: {len(result['candidates'])}")
        
        print(f"\n🏆 最终选择的 SQL:")
        print(f"```sql\n{result['sql']}\n```")
        
        print(f"\n📊 对比:")
        print(f"   真实 SQL: {sample.sql}")
        print(f"   生成 SQL: {result['sql']}")
    
    return result


def main():
    """主函数"""
    
    print("="*100)
    print("XiYan-SQL: 完整实现 - 端到端 SQL 生成")
    print("基于论文: XiYan-SQL: Simple and Effective Text-to-SQL with Schema Filter and Multi-Generator Ensemble")
    print("="*100)
    
    # 配置
    dataset_path = r"D:\download\dev_20240627"
    api_key = "sk-bb901ef8d7e44cb0be1c535e137974c4"
    model = "qwen3.6-plus"
    
    # 加载数据集
    print("\n📂 加载 BIRD 数据集...")
    dataset = BirdDataset(dataset_path)
    
    # 创建生成器
    print("\n🔧 初始化 XiYanSQLGenerator...")
    generator = XiYanSQLGenerator(api_key=api_key, model=model)
    
    # 交互式菜单
    while True:
        print(f"\n{'='*100}")
        print("选择操作:")
        print("  1. XiYan-SQL (完整架构: SchemaFilter + Multi-Generator + Reorganization + Selection)")
        print("  2. 仅 SchemaFilter + 单生成器")
        print("  3. 完整 Schema + 单生成器")
        print("  4. 批量演示前 N 个样本")
        print("  0. 退出")
        print(f"{'='*100}")
        
        choice = input("\n请输入选项 (0-4): ").strip()
        
        if choice == "0":
            print("👋 再见!")
            break
        
        elif choice == "1":
            idx = input(f"请输入样本索引 (0-{len(dataset.samples)-1}): ").strip()
            try:
                idx = int(idx)
                demo_single_sample(
                    generator, dataset, idx,
                    use_schema_filter=True, use_multi_generator=True
                )
            except ValueError:
                print("❌ 无效的索引!")
        
        elif choice == "2":
            idx = input(f"请输入样本索引 (0-{len(dataset.samples)-1}): ").strip()
            try:
                idx = int(idx)
                demo_single_sample(
                    generator, dataset, idx,
                    use_schema_filter=True, use_multi_generator=False
                )
            except ValueError:
                print("❌ 无效的索引!")
        
        elif choice == "3":
            idx = input(f"请输入样本索引 (0-{len(dataset.samples)-1}): ").strip()
            try:
                idx = int(idx)
                demo_single_sample(
                    generator, dataset, idx,
                    use_schema_filter=False, use_multi_generator=False
                )
            except ValueError:
                print("❌ 无效的索引!")
        
        elif choice == "4":
            n = input("请输入要演示的样本数量: ").strip()
            try:
                n = int(n)
                n = min(n, len(dataset.samples))
                
                results = []
                for i in range(n):
                    print(f"\n\n{'#'*100}")
                    print(f"处理样本 {i+1}/{n}")
                    print(f"{'#'*100}")
                    result = demo_single_sample(
                        generator, dataset, i,
                        use_schema_filter=True, use_multi_generator=True
                    )
                    results.append(result)
                
                print(f"\n\n{'='*100}")
                print(f"批量演示完成! 共处理 {n} 个样本")
                print(f"{'='*100}")
                
            except ValueError:
                print("❌ 无效的数量!")
        
        else:
            print("❌ 无效的选项!")


if __name__ == "__main__":
    main()
