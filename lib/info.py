class DatasetInfo:
    def __init__(self, name, url, mappings):
        self.name = name
        self.url = url
        self.mappings = mappings


def retinopathy_v2():
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

    mappings = {'c2': c2, 'c3': c3, 'c5': c5}
    url = "https://zenodo.org/record/4647952/files" \
          "/Dataset_from_fundus_images_for_the_study_of_diabetic_retinopathy_V02.zip?download=1 "
    name = 'retinopathy-v2'

    return DatasetInfo(name, url, mappings)
