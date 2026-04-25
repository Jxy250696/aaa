#!/usr/bin/env python3
"""
BIRD 数据集加载器
用于加载和处理 BIRD 数据集
"""

import os
import re
import json
import sqlite3
import csv
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from schema_filter import SchemaFilter, TableSchema


@dataclass
class BirdSample:
    """BIRD 数据样本"""
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    difficulty: str


class BirdDataset:
    """BIRD 数据集加载器"""
    
    def __init__(self, dataset_path: str):
        """
        初始化 BIRD 数据集
        
        Args:
            dataset_path: 数据集路径
        """
        self.dataset_path = dataset_path
        self.dev_json_path = os.path.join(dataset_path, "dev.json")
        self.databases_path = os.path.join(dataset_path, "dev_databases")
        
        self.samples: List[BirdSample] = []
        self.db_schemas: Dict[str, Dict[str, TableSchema]] = {}
        
        self._load_dataset()
        self._load_all_schemas()
    
    def _load_dataset(self):
        """加载 dev.json 数据集"""
        print(f"加载数据集: {self.dev_json_path}")
        
        with open(self.dev_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            sample = BirdSample(
                question_id=item["question_id"],
                db_id=item["db_id"],
                question=item["question"],
                evidence=item["evidence"],
                sql=item["SQL"],
                difficulty=item["difficulty"]
            )
            self.samples.append(sample)
        
        print(f"加载了 {len(self.samples)} 个样本")
    
    def _get_db_path(self, db_id: str) -> str:
        """
        获取数据库文件路径
        
        Args:
            db_id: 数据库 ID
            
        Returns:
            SQLite 数据库文件路径
        """
        return os.path.join(self.databases_path, db_id, f"{db_id}.sqlite")
    
    def _get_description_dir(self, db_id: str) -> str:
        """
        获取数据库描述文件目录路径
        
        Args:
            db_id: 数据库 ID
            
        Returns:
            描述文件目录路径
        """
        return os.path.join(self.databases_path, db_id, "database_description")
    
    def _load_column_descriptions(self, db_id: str) -> Dict[str, Dict[str, str]]:
        """
        加载数据库的列描述
        
        Args:
            db_id: 数据库 ID
            
        Returns:
            字典：表名 -> {列名 -> 描述}
        """
        descriptions = {}
        desc_dir = self._get_description_dir(db_id)
        
        if not os.path.exists(desc_dir):
            return descriptions
        
        # 尝试的编码列表（utf-8-sig 优先，它会自动去除 BOM 头）
        encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252', 'gbk', 'gb2312']
        
        # 遍历所有 CSV 文件
        for filename in os.listdir(desc_dir):
            if filename.endswith('.csv'):
                table_name = filename[:-4]  # 去掉 .csv 后缀
                csv_path = os.path.join(desc_dir, filename)
                
                col_descs = {}
                success = False
                
                # 尝试不同的编码
                for encoding in encodings:
                    try:
                        with open(csv_path, 'r', encoding=encoding) as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                # 规范化字典键名（去除可能的 BOM 和空白）
                                normalized_row = {}
                                for key, val in row.items():
                                    clean_key = key.strip()
                                    # 去除 BOM 字符（U+FEFF）
                                    if clean_key.startswith('\ufeff'):
                                        clean_key = clean_key[1:]
                                    normalized_row[clean_key] = val
                                
                                # 读取列信息
                                orig_col = normalized_row.get('original_column_name', '').strip()
                                col_name = normalized_row.get('column_name', '').strip()
                                col_desc = normalized_row.get('column_description', '').strip()
                                value_desc = normalized_row.get('value_description', '').strip()
                                
                                # 优先使用 original_column_name（这才是数据库里的真实字段名）
                                actual_col = orig_col if orig_col else col_name
                                if actual_col:
                                    # 组合描述（包括所有可用字段），并且把换行符替换为空格
                                    full_desc_parts = []
                                    if col_desc:
                                        # 把换行符和多余空格替换为单个空格
                                        clean_col_desc = re.sub(r'\s+', ' ', col_desc).strip()
                                        full_desc_parts.append(clean_col_desc)
                                    data_format_val = normalized_row.get('data_format', '').strip()
                                    if data_format_val:
                                        full_desc_parts.append(f"类型: {data_format_val}")
                                    if value_desc:
                                        # 把换行符和多余空格替换为单个空格
                                        clean_value_desc = re.sub(r'\s+', ' ', value_desc).strip()
                                        full_desc_parts.append(f"值说明: {clean_value_desc}")
                                    
                                    col_descs[actual_col] = " | ".join(full_desc_parts) if full_desc_parts else ""
                        
                        success = True
                        break
                        
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"  警告: 加载描述文件 {filename} (编码 {encoding}) 时出错: {e}")
                
                if success:
                    descriptions[table_name] = col_descs
                else:
                    print(f"  警告: 无法加载描述文件 {filename} (尝试了所有编码)")
        
        return descriptions
    
    def _load_schema_from_db(self, db_id: str) -> Dict[str, TableSchema]:
        """
        从 SQLite 数据库加载 schema
        
        Args:
            db_id: 数据库 ID
            
        Returns:
            表结构字典
        """
        db_path = self._get_db_path(db_id)
        
        if not os.path.exists(db_path):
            print(f"警告: 数据库文件不存在: {db_path}")
            return {}
        
        # 先加载列描述
        column_descriptions = self._load_column_descriptions(db_id)
        
        schema = {}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 获取所有表名
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                # 跳过 SQLite 系统表（如 sqlite_sequence, sqlite_stat1 等）
                if table_name.startswith('sqlite_'):
                    continue
                
                try:
                    # 获取表的列信息（使用参数化避免 SQL 注入）
                    # 对于 SQLite，PRAGMA table_info 需要字符串
                    cursor.execute(f'PRAGMA table_info("{table_name}");')
                    columns_info = cursor.fetchall()
                    
                    columns = []
                    primary_key = None
                    
                    for col_info in columns_info:
                        col_name = col_info[1]
                        columns.append(col_name)
                        
                        # 检查是否是主键
                        if col_info[5] == 1:  # pk 列
                            primary_key = col_name
                    
                    # 获取该表的列描述，并只保留实际存在的列
                    table_col_descs_all = column_descriptions.get(table_name, {})
                    table_col_descs = {}
                    
                    print(f"\n  匹配表 {db_id}.{table_name} 的列描述:")
                    print(f"  数据库列数: {len(columns)}, CSV 描述数: {len(table_col_descs_all)}")
                    
                    # 构建一个标准化的列名映射
                    normalized_csv_cols = {}  # 标准化后列名 -> 原始 CSV 列名
                    for csv_col in table_col_descs_all.keys():
                        norm_key = csv_col.strip().lower()
                        normalized_csv_cols[norm_key] = csv_col
                    
                    for col_name in columns:
                        # 先精确匹配
                        found = False
                        
                        # 1. 精确匹配
                        if col_name in table_col_descs_all:
                            table_col_descs[col_name] = table_col_descs_all[col_name]
                            found = True
                            # print(f"    ✓ 精确匹配: {col_name}")
                        else:
                            # 2. 标准化后匹配
                            col_norm = col_name.strip().lower()
                            if col_norm in normalized_csv_cols:
                                orig_csv_col = normalized_csv_cols[col_norm]
                                table_col_descs[col_name] = table_col_descs_all[orig_csv_col]
                                found = True
                                print(f"    ✓ 标准化匹配: {col_name} (来自 CSV: {orig_csv_col})")
                        
                        if not found:
                            print(f"    ✗ 未找到描述: {col_name}")
                    
                    # 调试信息：检查是否有描述但实际不存在的列
                    extra_cols = set(table_col_descs_all.keys()) - set(columns)
                    if extra_cols:
                        print(f"  ℹ️  表 {db_id}.{table_name}: CSV 有描述但数据库不存在的列: {sorted(extra_cols)}")
                        
                    print(f"  成功匹配 {len(table_col_descs)} 个列描述")
                    
                    schema[table_name] = TableSchema(
                        name=table_name,
                        columns=columns,
                        column_descriptions=table_col_descs,
                        primary_key=primary_key
                    )
                except Exception as e:
                    print(f"  警告: 加载表 {db_id}.{table_name} 时出错: {e}")
            
            conn.close()
            
        except Exception as e:
            print(f"加载数据库 {db_id} 时出错: {e}")
        
        return schema
    
    def _load_all_schemas(self):
        """加载所有数据库的 schema"""
        print("\n加载数据库 schema...")
        
        db_ids = set(sample.db_id for sample in self.samples)
        
        for db_id in db_ids:
            print(f"  加载 {db_id}...")
            self.db_schemas[db_id] = self._load_schema_from_db(db_id)
        
        print(f"加载了 {len(self.db_schemas)} 个数据库的 schema")
    
    def get_sample(self, index: int) -> BirdSample:
        """
        获取指定索引的样本
        
        Args:
            index: 样本索引
            
        Returns:
            BirdSample 对象
        """
        return self.samples[index]
    
    def get_samples_by_db(self, db_id: str) -> List[BirdSample]:
        """
        获取指定数据库的所有样本
        
        Args:
            db_id: 数据库 ID
            
        Returns:
            BirdSample 列表
        """
        return [sample for sample in self.samples if sample.db_id == db_id]
    
    def get_schema(self, db_id: str) -> Dict[str, TableSchema]:
        """
        获取指定数据库的 schema
        
        Args:
            db_id: 数据库 ID
            
        Returns:
            表结构字典
        """
        return self.db_schemas.get(db_id, {})


class BirdSchemaFilterDemo:
    """BIRD 数据集上的 SchemaFilter 演示"""
    
    def __init__(self, dataset: BirdDataset):
        """
        初始化演示
        
        Args:
            dataset: BIRD 数据集
        """
        self.dataset = dataset
        self.filter = SchemaFilter()
    
    def demo_sample(self, sample_index: int):
        """
        演示单个样本的 SchemaFilter
        
        Args:
            sample_index: 样本索引
        """
        sample = self.dataset.get_sample(sample_index)
        
        print(f"\n{'='*80}")
        print(f"样本 #{sample.question_id}")
        print(f"数据库: {sample.db_id}")
        print(f"难度: {sample.difficulty}")
        print(f"{'='*80}")
        print(f"\n问题: {sample.question}")
        print(f"\n证据: {sample.evidence if sample.evidence else '(无)'}")
        print(f"\n真实 SQL: {sample.sql}")
        
        # 获取完整 schema
        full_schema = self.dataset.get_schema(sample.db_id)
        
        print(f"\n完整 Schema（带描述）:")
        for table_name, table in full_schema.items():
            print(f"\n  表: {table_name}")
            for col in table.columns:
                desc = table.get_column_description(col)
                if desc:
                    print(f"    - {col}: {desc}")
                else:
                    print(f"    - {col}")
        
        # 应用 SchemaFilter
        filtered = self.filter.filter_schema(
            question=sample.question,
            schema=full_schema
        )
        
        # 生成提示词（带描述）
        prompt = self.filter.schema_to_prompt(filtered, include_descriptions=True)
        print(f"\n{'='*80}")
        print(f"过滤后的 Schema 提示词（带描述）:")
        print(f"{'='*80}")
        print(prompt)
        
        return filtered


def main():
    """主函数"""
    
    # 数据集路径
    dataset_path = "/cpfs01/projects-HDD/cfff-8f1c54eecb30_HDD/public/evaluate_nl2sql/dataset/dev_20240627"
    
    # 加载数据集
    dataset = BirdDataset(dataset_path)
    
    # 创建演示
    demo = BirdSchemaFilterDemo(dataset)
    
    # 演示前几个样本
    print("\n" + "="*80)
    print("SchemaFilter 在 BIRD 数据集上的演示")
    print("="*80)
    
    for i in range(min(3, len(dataset.samples))):
        demo.demo_sample(i)
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
