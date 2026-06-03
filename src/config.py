from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
CARS_DIR = DATA_DIR / "cars"
FLICKR8K_DIR = DATA_DIR / "flickr8k"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DESCRIPTORS_DIR = ARTIFACTS_DIR / "descriptors"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

GROUP_ID = 8

GROUP_08_QUERIES = [
    ("R1", 1, "1_4_Kia_stinger_1990"),
    ("R2", 1, "1_2_Kia_sorento_1675"),
    ("R3", 1, "1_9_Kia_stonic_2677"),

    ("R4", 3, "3_1_Renault_Twingo_4487"),
    ("R5", 3, "3_0_Renault_grandscenic_4372"),
    ("R6", 3, "3_5_Renault_clio_5101"),

    ("R7", 5, "5_0_Mercedes_ClasseCLS_7059"),
    ("R8", 5, "5_4_Mercedes_GLEcoupe_7428"),
    ("R9", 5, "5_8_Mercedes_CLA_7992"),

    ("R10", 7, "7_0_Peugeot_508break_9591"),
    ("R11", 7, "7_3_Peugeot_Rifter_10091"),
    ("R12", 7, "7_6_Peugeot_3008_10530"),

    ("R13", 9, "9_0_Audi_A6_12268"),
    ("R14", 9, "9_3_Audi_Q7_12722"),
    ("R15", 9, "9_4_Audi_A1_12910"),
]