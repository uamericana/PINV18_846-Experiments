from enum import Enum
from abc import ABC


class DatasetInfo(ABC):
    name: str
    url: str
    mapping: Enum
