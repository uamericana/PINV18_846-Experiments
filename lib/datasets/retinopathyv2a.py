from lib.datasets.info import DatasetMapping, DatasetInfo

_fondo_normal = 'fondo-normal-anon'


class RetinopathyV2aMapping(DatasetMapping):
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


class RetinopathyV2a(DatasetInfo):
    name = 'retinopathy-v2a'
    url = ""
    mapping = RetinopathyV2aMapping
