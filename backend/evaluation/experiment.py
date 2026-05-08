"""
Experiment tracking — what was tested, with what settings.

Every evaluation run is tied to an ExperimentConfig so results 
are always reproducible and comparable.
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    """Captures the full context of an extraction experiment."""
    
    # What
    experiment_name: str
    domain: str                         # "process_knowledge", "organizational_knowledge"
    schema_level: int                   # 1-5 complexity level
    
    # How
    model: str                          # "llama3.2", "llama3.1:70b", "custom-finetuned-v1"
    prompt_strategy: str                # "default", "few_shot", "custom_v3", "chain_of_thought"
    temperature: float = 0.0
    
    # Optional context
    notes: Optional[str] = None
    timestamp: Optional[datetime] = None
    chunking_strategy: Optional[str] = None    # "none", "fixed_500", "semantic"
    finetuning_dataset: Optional[str] = None   # if using a fine-tuned model
    num_few_shot_examples: Optional[int] = None


class ExperimentResult(BaseModel):
    """Bundles config + metrics for one experiment run."""
    
    config: ExperimentConfig
    entity_metrics: Optional[dict] = None      # filled by entity_eval
    attribute_metrics: Optional[dict] = None   # filled by attribute_eval  
    relation_metrics: Optional[dict] = None    # filled by relation_eval
    structure_metrics: Optional[dict] = None   # filled by structure_eval