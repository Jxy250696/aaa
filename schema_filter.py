#!/usr/bin/env python3
"""
Schema Filter 实现（论文版）
根据用户问题和证据，过滤和选择相关的数据库 schema 元素
"""

import os
import re
from typing import Dict, List, Tuple, Set, Any
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class TableSchema:
    """表结构"""
    name: str
    columns: List[str]
    column_descriptions: Dict[str, str] = None  # 列名 -> 描述
    table_description: str = None  # 表描述
    primary_key: str = None
    foreign_keys: Dict[str, Tuple[str, str]] = None
    
    def __post_init__(self):
        if self.foreign_keys is None:
            self.foreign_keys = {}
        if self.column_descriptions is None:
            self.column_descriptions = {}
    
    def get_column_description(self, column_name: str) -> str:
        """获取列的描述，去除换行符并返回，如果没有则返回空字符串"""
        desc = self.column_descriptions.get(column_name, "")
        # 把换行符和多余空格替换为单个空格
        return re.sub(r'\s+', ' ', desc).strip()


@dataclass
class FilteredSchema:
    """过滤后的 schema（单个版本）"""
    tables: List[TableSchema]
    relevant_columns: Dict[str, List[str]]  # 表名 -> 相关列
    confidence_scores: Dict[str, float]  # 表名 -> 置信度


@dataclass
class FilteredSchemaSet:
    """多个 schema 版本的集合（对应论文中的 S = {S_1, ..., S_p_s}）"""
    schemas: List[FilteredSchema]
    best_schema_index: int = 0  # 哪个是"最佳"的（默认第一个）
    
    @property
    def best_schema(self) -> FilteredSchema:
        """获取最佳 schema"""
        return self.schemas[self.best_schema_index] if self.schemas else None


class SchemaFilter:
    """Schema 过滤器（论文版 Algorithm 1）"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        初始化 SchemaFilter
        
        Args:
            api_key: 阿里云 API Key
            base_url: API Base URL
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "sk-bb901ef8d7e44cb0be1c535e137974c4")
        self.base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = None
        
        # 尝试初始化 OpenAI 客户端
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            print("SchemaFilter initialized successfully")
        except Exception as e:
            print(f"SchemaFilter initialization warning: {e}")
            print("Will use simple keyword-based method")
    
    def _extract_keywords_with_llm(self, question: str, evidence: str = "") -> List[str]:
        """
        使用大模型从问题和证据中提取关键词
        
        Args:
            question: 用户问题
            evidence: 证据信息
            
        Returns:
            关键词列表
        """
        if not self.client:
            return self._simple_extract_keywords(question)
        
        try:
            prompt = f"""You are a helpful assistant that extracts key terms from a user's question and evidence for Text-to-SQL task.

Question: {question}
{("Evidence: " + evidence) if evidence else ""}

Please extract 5-10 key terms that are most relevant to potential database tables and columns. Focus on:
- Entities mentioned (people, places, things)
- Attributes (properties, characteristics)
- Actions or operations
- Any numeric values or dates

Provide only the keywords separated by commas, without any explanation or numbering."""

            response = self.client.chat.completions.create(
                model="qwen3.6-plus",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            keywords = []
            parts = re.split(r'[,，\n]', content)
            for part in parts:
                part = part.strip()
                part = re.sub(r'^\d+\.\s*', '', part)
                part = re.sub(r'^[-*•]\s*', '', part)
                if part:
                    keywords.append(part)
            
            return keywords if keywords else self._simple_extract_keywords(question)
            
        except Exception as e:
            print(f"Keyword extraction with LLM failed: {e}")
            return self._simple_extract_keywords(question)
    
    def _simple_extract_keywords(self, text: str) -> List[str]:
        """简单的关键词提取方法（备用方案）"""
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
            'just', 'don', 'now'
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 1]
        return keywords
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        if self.client:
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-v4",
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Failed to get embedding: {e}")
        
        return self._simple_embedding(text)
    
    def _simple_embedding(self, text: str) -> List[float]:
        """简单的伪嵌入方法（用于演示）"""
        import hashlib
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        embedding = []
        for i in range(256):
            byte_idx = i % len(hash_bytes)
            val = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(val)
        return embedding
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _flatten_columns(self, schema: Dict[str, TableSchema]) -> List[Tuple[str, str]]:
        """
        将所有表的所有列展平为一个列表
        
        Args:
            schema: 完整的 schema
            
        Returns:
            [(table_name, column_name), ...]
        """
        all_columns = []
        for table_name, table in schema.items():
            for column_name in table.columns:
                all_columns.append((table_name, column_name))
        return all_columns
    
    def _identify_pk_fk_columns(
        self,
        selected_table_columns: List[Tuple[str, str]],
        full_schema: Dict[str, TableSchema]
    ) -> List[Tuple[str, str]]:
        """
        从选中的列中识别出 PK/FK 列（Paper: PFKeyIdentifier）
        
        Args:
            selected_table_columns: 当前选中的列 [(table, column), ...]
            full_schema: 完整的 schema
            
        Returns:
            被识别为 PK/FK 的列
        """
        pk_fk_columns = []
        
        for table_name, column_name in selected_table_columns:
            if table_name not in full_schema:
                continue
            table = full_schema[table_name]
            
            # 检查是否是主键
            if table.primary_key == column_name:
                pk_fk_columns.append((table_name, column_name))
            
            # 检查是否是外键
            if column_name in table.foreign_keys:
                pk_fk_columns.append((table_name, column_name))
        
        return pk_fk_columns
    
    def _select_columns_by_similarity(
        self,
        candidate_columns: List[Tuple[str, str]],
        question: str,
        evidence: str,
        full_schema: Dict[str, TableSchema],
        select_top_k: int = None
    ) -> List[Tuple[str, str]]:
        """
        根据相似度选择相关列（Paper: f_M_s）
        
        Args:
            candidate_columns: 候选列 [(table, column), ...]
            question: 用户问题
            evidence: 证据信息
            full_schema: 完整的 schema
            select_top_k: 要选择的数量（None 表示按阈值）
            
        Returns:
            选中的列 [(table, column), ...]
        """
        # 合并问题和证据
        q_plus_e = question
        if evidence:
            q_plus_e = f"{question} {evidence}"
        
        q_e_embedding = self._get_embedding(q_plus_e)
        
        # 计算每个列的相似度
        column_scores = []
        for table_name, column_name in candidate_columns:
            table = full_schema[table_name]
            col_desc = table.get_column_description(column_name)
            if col_desc:
                column_text = f"{table_name}.{column_name}: {col_desc}"
            else:
                column_text = f"{table_name}.{column_name}"
            
            column_embedding = self._get_embedding(column_text)
            similarity = self._cosine_similarity(q_e_embedding, column_embedding)
            column_scores.append(((table_name, column_name), similarity))
        
        # 按分数排序
        column_scores.sort(key=lambda x: -x[1])
        
        # 选择列
        if select_top_k is not None:
            selected_columns = [col for col, score in column_scores[:select_top_k]]
        else:
            # 如果没有指定 K，选择分数 >= 0.05 的
            selected_columns = [col for col, score in column_scores if score >= 0.05]
        
        return selected_columns
    
    def _build_filtered_schema(
        self,
        selected_columns: List[Tuple[str, str]],
        full_schema: Dict[str, TableSchema]
    ) -> FilteredSchema:
        """
        根据选中的列构建 FilteredSchema 对象
        
        Args:
            selected_columns: 选中的列 [(table, column), ...]
            full_schema: 完整的 schema
            
        Returns:
            FilteredSchema 对象
        """
        # 确定涉及的表
        table_names = set()
        for table_name, _ in selected_columns:
            table_names.add(table_name)
        
        # 构建表对象
        tables = []
        for table_name in table_names:
            if table_name in full_schema:
                tables.append(full_schema[table_name])
        
        # 组织列信息
        relevant_columns = {}
        for table in tables:
            table_name = table.name
            cols = [col_name for t_name, col_name in selected_columns if t_name == table_name]
            relevant_columns[table_name] = cols
        
        # 构建结果
        result = FilteredSchema(
            tables=tables,
            relevant_columns=relevant_columns,
            confidence_scores={}
        )
        
        return result
    
    def filter_schema(
        self,
        question: str,
        schema: Dict[str, TableSchema],
        evidence: str = "",
        max_iterations: int = 3  # Paper: p_s
    ) -> FilteredSchemaSet:
        """
        过滤 schema（论文版 Algorithm 1 实现）
        
        Args:
            question: 用户问题
            schema: 完整的数据库 schema
            evidence: 证据信息
            max_iterations: 最大迭代次数（即要生成的 schema 版本数量）
            
        Returns:
            FilteredSchemaSet（包含多个 schema 版本）
        """
        print(f"\n{'='*60}")
        print("Schema Filter Processing (Paper Algorithm 1)")
        print(f"Question: {question}")
        if evidence:
            print(f"Evidence: {evidence}")
        print(f"Max iterations (p_s): {max_iterations}")
        print(f"{'='*60}")
        
        # 论文步骤 1: 初始化 S
        schema_set_list = []
        
        # 论文步骤 2: 展平所有列，准备 S^trv (候选池)
        S_trv = self._flatten_columns(schema)
        print(f"\nInitial candidate columns (S^trv): {len(S_trv)} columns")
        
        # 存储之前所有迭代选中的列
        cumulative_columns = []
        
        # 论文步骤 2: 迭代
        for i in range(1, max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"Iteration {i}/{max_iterations}")
            print(f"{'='*60}")
            
            # 论文步骤 3: 从 S^trv 选择相关列 (f_M_s)
            # 每次选的列数递减，或者至少选 3 列
            select_k = max(3, len(S_trv) // (max_iterations - i + 1))
            
            S_slct = self._select_columns_by_similarity(
                candidate_columns=S_trv,
                question=question,
                evidence=evidence,
                full_schema=schema,
                select_top_k=select_k
            )
            print(f"Selected columns (S^slct): {len(S_slct)} columns")
            
            # 论文步骤 4: 识别 PK/FK (PFKeyIdentifier)
            P_i = self._identify_pk_fk_columns(S_slct, schema)
            print(f"Identified PK/FK columns (P_i): {len(P_i)} columns")
            
            # 论文步骤 5: 合并，得到 S_i
            # S_i = (∪_{k=1}^{i-1} S_k) ∪ S^slct ∪ P_i
            S_i_columns = cumulative_columns.copy()
            S_i_columns.extend(S_slct)
            S_i_columns.extend(P_i)
            # 去重
            S_i_columns = list(dict.fromkeys(S_i_columns))
            
            S_i = self._build_filtered_schema(S_i_columns, schema)
            print(f"Generated schema S_{i}: {len(S_i_columns)} columns, {len(S_i.tables)} tables")
            
            # 论文步骤 6: 添加到结果集合
            schema_set_list.append(S_i)
            
            # 更新累积列
            cumulative_columns = S_i_columns.copy()
            
            # 论文步骤 7: 更新候选池 S^trv
            # 移除已选的非键列，保留键列
            # 先找出 S^slct 中的非键列
            key_columns_set = set(P_i)
            columns_to_remove = [col for col in S_slct if col not in key_columns_set]
            # 从 S^trv 中移除这些非键列
            S_trv = [col for col in S_trv if col not in columns_to_remove]
            
            print(f"Updated candidate pool (S^trv): {len(S_trv)} columns left")
        
        # 论文步骤 8: 返回 S
        result = FilteredSchemaSet(
            schemas=schema_set_list,
            best_schema_index=0  # 默认第一个是最佳的，或者我们也可以让最后一个是最佳的
        )
        
        print(f"\n{'='*60}")
        print(f"Schema Filter Completed! Generated {len(schema_set_list)} schema versions.")
        print(f"{'='*60}")
        
        return result
    
    def schema_to_prompt(self, filtered_schema: FilteredSchema, include_descriptions: bool = True) -> str:
        """
        将过滤后的 schema 转换为提示词格式
        
        Args:
            filtered_schema: 过滤后的 schema
            include_descriptions: 是否包含列描述
            
        Returns:
            提示词文本
        """
        parts = ["Database Schema:"]
        
        for table in filtered_schema.tables:
            table_name = table.name
            cols = filtered_schema.relevant_columns.get(table_name, table.columns)
            
            parts.append(f"\nTable: {table_name}")
            
            col_lines = []
            for col in cols:
                desc = table.get_column_description(col)
                if include_descriptions and desc:
                    # 最后再确保一次，把所有换行符和多余空格替换成单个空格
                    clean_desc = re.sub(r'\s+', ' ', desc).strip()
                    col_lines.append(f"  - {col}: {clean_desc}")
                else:
                    col_lines.append(f"  - {col}")
            
            parts.append("Columns:\n" + "\n".join(col_lines))
            
            if table.primary_key and table.primary_key in cols:
                parts.append(f"Primary Key: {table.primary_key}")
            
            if table.foreign_keys:
                for col, (ref_table, ref_col) in table.foreign_keys.items():
                    if col in cols:
                        parts.append(f"Foreign Key: {col} -> {ref_table}.{ref_col}")
        
        return "\n".join(parts)


def main():
    """测试 SchemaFilter"""
    
    sample_schema = {
        "employees": TableSchema(
            name="employees",
            columns=["id", "name", "age", "department_id", "salary", "hire_date"],
            primary_key="id",
            foreign_keys={"department_id": ("departments", "id")}
        ),
        "departments": TableSchema(
            name="departments",
            columns=["id", "name", "location", "manager_id"],
            primary_key="id"
        ),
        "products": TableSchema(
            name="products",
            columns=["id", "name", "price", "category", "stock"],
            primary_key="id"
        ),
        "orders": TableSchema(
            name="orders",
            columns=["id", "product_id", "employee_id", "quantity", "order_date"],
            primary_key="id",
            foreign_keys={
                "product_id": ("products", "id"),
                "employee_id": ("employees", "id")
            }
        )
    }
    
    filter = SchemaFilter()
    
    test_questions = [
        ("Show names and ages of all employees", ""),
    ]
    
    for question, evidence in test_questions:
        filtered_set = filter.filter_schema(question, sample_schema, evidence, max_iterations=3)
        
        for idx, schema in enumerate(filtered_set.schemas):
            print(f"\n{'='*80}")
            print(f"Schema version {idx+1}")
            print(f"{'='*80}")
            prompt = filter.schema_to_prompt(schema)
            print(prompt)


if __name__ == "__main__":
    main()
