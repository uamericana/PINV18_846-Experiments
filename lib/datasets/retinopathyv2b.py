from enum import Enum

from lib.datasets.info import DatasetInfo

_fondo_normal = 'fondo-normal-anon'


class RetinopathyV2bMapping(Enum):
    c2 = {
        "No DR Signs": ["No DR Signs", _fondo_normal],
        "Has DR Signs": [
            "Mild (or early) NPDR",
            "Moderate NPDR",
            "Severe NPDR",
            "Very Severe NPDR",
            "PDR",
            "Advanced PDR",
        ],
    }

    c3 = {
        "No DR Signs": ["No DR Signs", _fondo_normal],
        "NPDR": [
            "Mild (or early) NPDR",
            "Moderate NPDR",
            "Severe NPDR",
            "Very Severe NPDR",
        ],
        "PDR": [
            "PDR",
            "Advanced PDR",
        ],
    }

    c5 = {
        "No DR Signs": ["No DR Signs", _fondo_normal],
        "Mild (or early) NPDR": ["Mild (or early) NPDR"],
        "Moderate NPDR": ["Moderate NPDR"],
        "Severe NPDR": ["Severe NPDR", "Very Severe NPDR"],
        "PDR": ["PDR", "Advanced PDR"],
    }


class RetinopathyV2b(DatasetInfo):
    name = 'retinopathy-v2b'
    url = ""
    mapping = RetinopathyV2bMapping
