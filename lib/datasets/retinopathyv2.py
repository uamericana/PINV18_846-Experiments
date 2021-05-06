from lib.datasets.info import DatasetMapping, DatasetInfo


class RetinopathyV2Mapping(DatasetMapping):
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


class RetinopathyV2(DatasetInfo):
    name = 'retinopathy-v2'
    url = "https://zenodo.org/record/4647952/files" \
          "/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02.zip?download=1"
    mapping = RetinopathyV2Mapping
