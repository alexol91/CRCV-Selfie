dataset_file_name = "./Selfie-dataset/selfie_dataset.txt"
dataset_image_dir = "./Selfie-dataset/images/"

IMAGE_DIM = (64, 64)
# DATASET COLUMNS
col = {
    'imageName': 0,
    'popularityScore': 1,
    'partialFaces': 2,
    'female': 3,
    'baby': 4,
    'child': 5,
    'teenager': 6,
    'youth': 7,
    'middleAge': 8,
    'senior': 9,
    'white': 10,
    'black': 11,
    'asian': 12,
    'ovalFace': 13,
    'roundFace': 14,
    'heartFace': 15,
    'smiling': 16,
    'mouthOpen': 17,
    'frowning': 18,
    'wearingGlasses': 19,
    'wearingSunglasses': 20,
    'wearingLipstick': 21,
    'tongueOut': 22,
    'duckFace': 23,
    'blackHair': 24,
    'blondHair': 25,
    'brownHair': 26,
    'redHair': 27,
    'curlyHair': 28,
    'straightHair': 29,
    'braidHair': 30,
    'showingCellphone': 31,
    'usingEarphone': 32,
    'usingMirror': 33,
    'braces': 34,
    'wearingHat': 35,
    'harshLighting': 36,
    'dimLighting': 37
}

best_models = {
    1: "weights/weights.1.01.h5",
    2: "weights/weights.2.01.h5",
    3: "weights/weights.2.01.h5",
    4: "weights/weights.4.01.h5",
    5: "weights/weights.5.01.h5",
    6: "weights/weights.6.01.h5",
}