from dataclasses import dataclass
from transformers import PreTrainedModel

@dataclass
class TransformerConfig:
    processor_class: PreTrainedModel
    processor_name: str
    model_class: PreTrainedModel
    model_name: str
