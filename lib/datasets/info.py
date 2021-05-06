from enum import Enum
from abc import ABC, ABCMeta


class DatasetMapping(ABC):
    pass


class DatasetInfo(ABC):
    name: str
    url: str
    mapping: DatasetMapping
