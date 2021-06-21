from enum import Enum

from lib.datasets.info import DatasetInfo


class RetinopathyV3Mapping(Enum):
    c2 = {
        "No DR Signs": ["No DR Signs"],
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
        "No DR Signs": ["No DR Signs"],
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
        "No DR Signs": ["No DR Signs"],
        "Mild (or early) NPDR": ["Mild (or early) NPDR"],
        "Moderate NPDR": ["Moderate NPDR"],
        "Severe NPDR": ["Severe NPDR", "Very Severe NPDR"],
        "PDR": ["PDR", "Advanced PDR"],
    }


class RetinopathyV3(DatasetInfo):
    name = 'retinopathy-v3'
    url = "https://zenodo.org/record/4891308/files/Dataset%20from%20fundus%20images%20for%20the%20study%20of" \
          "%20diabetic%20retinopathy_V03.zip?download=1 "
    mapping = RetinopathyV3Mapping
