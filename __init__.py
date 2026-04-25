
"""
XiYan-SQL: A Novel Multi-Generator Framework For Text-to-SQL
"""

__version__ = "1.0.0"

from .xiyan_sql import (
    XiYanSQL,
    SQLComponents,
    SchemaEncoder,
    ComponentGenerator,
    GeneratorCoordinator
)



from .trainer import (
    XiYanSQLTrainer,
    XiYanSQLInference,
    EvaluationMetrics
)

from .config import (
    XiYanSQLConfig,
    ModelConfig,
    TrainingConfig,
    InferenceConfig
)

from .advanced import (
    XiYanSQLHuggingFace,
    BeamSearchGenerator,
    SelfConsistencyValidator
)

__all__ = [
    "XiYanSQL",
    "SQLComponents",
    "SchemaEncoder",
    "ComponentGenerator",
    "GeneratorCoordinator",
    "SchemaParser",
    "SQLParser",
    "NL2SQLDataset",
    "XiYanSQLTrainer",
    "XiYanSQLInference",
    "EvaluationMetrics",
    "XiYanSQLConfig",
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "XiYanSQLHuggingFace",
    "BeamSearchGenerator",
    "SelfConsistencyValidator"
]
