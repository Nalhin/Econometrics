TRANSLATIONS = {
    "mean": "Średnia",
    "std": "Wariacja",
    "min": "Minimum",
    "25%": "Q1",
    "50%": "Mediana",
    "75%": "Q3",
    "max": "Maximum",
    "skewness": "Skośność",
    "kurtosis": "Kurtoza",
}


def apply_translations(description):
    for key in description.keys():
        description[TRANSLATIONS[key]] = description.pop(key)
    return description
