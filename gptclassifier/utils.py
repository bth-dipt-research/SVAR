from pydantic import BaseModel
from enum import Enum
from pathlib import Path

class Target(str, Enum):
    process = 'Process'
    product = 'Product'
    documentation = 'Documentation'

class Nature(str, Enum):
    quantitative = 'Quantitative'
    qualitative = 'Qualitative'
    mixed = 'Mixed'

class Interpretability(str, Enum):
    nonambiguous = 'Non-ambiguous'
    natural = 'Ambiguous (natural)'
    artificial = 'Ambiguous (artificial)'

class Reference(str, Enum):
    noreference = 'No reference'
    local = 'Local'
    internal = 'Internal'
    external = 'External'

class Classification(BaseModel):
    target: Target
    nature: Nature
    interpretability: Interpretability
    reference: Reference

class Prediction(BaseModel):
    id: str
    requirement: str
    classification: Classification

