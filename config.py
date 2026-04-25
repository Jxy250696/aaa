"""
XiYan-SQL 配置管理
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class APIConfig:
    """API 配置"""
    api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "sk-bb901ef8d7e44cb0be1c535e137974c4"))
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen3.6-plus"
    selection_model: str = "qwen3.6-plus"
    embedding_model: str = "text-embedding-v4"


@dataclass
class DatabaseConfig:
    """数据库配置"""
    dataset_path: str = r"D:\download\dev_20240627"
    db_base_path: str = "dev_databases"
    dialect: str = "sqlite"
    
    def get_db_path(self, db_id: str) -> str:
        """获取指定数据库的完整路径"""
        return os.path.join(self.dataset_path, self.db_base_path, db_id, f"{db_id}.sqlite")


@dataclass
class SchemaFilterConfig:
    """Schema Filter 配置"""
    enabled: bool = True
    max_iterations: int = 2  # 论文中的 p_s


@dataclass
class GeneratorConfig:
    """生成器配置"""
    enabled: bool = True
    # 生成器列表: (名称, 温度)
    generators: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("standard", 0.1),      # 标准风格
        ("detailed", 0.2),      # 详细思考
        ("simple", 0.1),        # 简洁风格
        ("structural", 0.3),    # 结构变体（CTE等）
        ("icl", 0.1),           # ICL 风格（few-shot 示例）
    ])


@dataclass
class ReorganizerConfig:
    """候选重组配置"""
    generator_order: Optional[List[int]] = None  # 生成器性能排序，None 表示按 ID 排序


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: str = "./predictions"
    prompt_output_dir: str = "./prompts"
    save_prompts: bool = True
    output_filename_xiyan: str = "predict_dev_xiyan.json"
    output_filename_filtered: str = "predict_dev_filtered.json"
    output_filename_full: str = "predict_dev_full.json"


@dataclass
class BatchConfig:
    """批量生成配置"""
    num_samples: Optional[int] = None  # None 表示全部
    start_index: int = 0


class XiYanSQLConfig:
    """XiYan-SQL 总配置"""
    
    def __init__(self):
        self.api = APIConfig()
        self.database = DatabaseConfig()
        self.schema_filter = SchemaFilterConfig()
        self.generator = GeneratorConfig()
        self.reorganizer = ReorganizerConfig()
        self.output = OutputConfig()
        self.batch = BatchConfig()
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'XiYanSQLConfig':
        """从字典创建配置"""
        config = cls()
        if 'api' in config_dict:
            config.api = APIConfig(**config_dict['api'])
        if 'database' in config_dict:
            config.database = DatabaseConfig(**config_dict['database'])
        if 'schema_filter' in config_dict:
            config.schema_filter = SchemaFilterConfig(**config_dict['schema_filter'])
        if 'generator' in config_dict:
            config.generator = GeneratorConfig(**config_dict['generator'])
        if 'reorganizer' in config_dict:
            config.reorganizer = ReorganizerConfig(**config_dict['reorganizer'])
        if 'output' in config_dict:
            config.output = OutputConfig(**config_dict['output'])
        if 'batch' in config_dict:
            config.batch = BatchConfig(**config_dict['batch'])
        return config
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'api': {
                'api_key': self.api.api_key,
                'base_url': self.api.base_url,
                'model': self.api.model,
                'selection_model': self.api.selection_model,
            },
            'database': {
                'dataset_path': self.database.dataset_path,
                'db_base_path': self.database.db_base_path,
                'dialect': self.database.dialect,
            },
            'schema_filter': {
                'enabled': self.schema_filter.enabled,
                'max_iterations': self.schema_filter.max_iterations,
            },
            'generator': {
                'enabled': self.generator.enabled,
                'generators': self.generator.generators,
            },
            'reorganizer': {
                'generator_order': self.reorganizer.generator_order,
            },
            'output': {
                'output_dir': self.output.output_dir,
                'prompt_output_dir': self.output.prompt_output_dir,
                'save_prompts': self.output.save_prompts,
                'output_filename_xiyan': self.output.output_filename_xiyan,
                'output_filename_filtered': self.output.output_filename_filtered,
                'output_filename_full': self.output.output_filename_full,
            },
            'batch': {
                'num_samples': self.batch.num_samples,
                'start_index': self.batch.start_index,
            }
        }
