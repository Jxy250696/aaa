
"""
Configuration for XiYan-SQL
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    hidden_size: int = 512
    num_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 512


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    device: str = "cuda"


@dataclass
class InferenceConfig:
    beam_size: int = 5
    max_generation_length: int = 128
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    num_samples: int = 5


class XiYanSQLConfig:
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.inference = InferenceConfig()

    @classmethod
    def from_dict(cls, config_dict: dict) -&gt; 'XiYanSQLConfig':
        config = cls()
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'inference' in config_dict:
            config.inference = InferenceConfig(**config_dict['inference'])
        return config

    def to_dict(self) -&gt; dict:
        return {
            'model': {
                'vocab_size': self.model.vocab_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'dropout': self.model.dropout,
                'max_sequence_length': self.model.max_sequence_length
            },
            'training': {
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'weight_decay': self.training.weight_decay,
                'num_epochs': self.training.num_epochs,
                'warmup_steps': self.training.warmup_steps,
                'gradient_clip': self.training.gradient_clip,
                'device': self.training.device
            },
            'inference': {
                'beam_size': self.inference.beam_size,
                'max_generation_length': self.inference.max_generation_length,
                'temperature': self.inference.temperature,
                'top_k': self.inference.top_k,
                'top_p': self.inference.top_p,
                'num_samples': self.inference.num_samples
            }
        }
