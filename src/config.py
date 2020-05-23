BERLIN_CENTER = (52.5027778, 13.404166666666667)

MODEL_COLUMNS = [
    "Id",
    "Price",
    "NumberOfReviews",
    "Availability365",
    "MinimumNights",
    "Bedrooms",
    "Bathrooms",
    "RoomType",
    "Latitude",
    "Longitude",
    "Accommodates",
]

CORR_MATRIX_COLUMNS = [
    "Price",
    "NumberOfReviews",
    "Availability365",
    "MinimumNights",
    "Bedrooms",
    "Bathrooms",
    "Accommodates",
    "DistanceFromCenter",
]

OUTPUT_PATH = "latex/generated"

P_VALUE_THRESHOLD = 0.05
