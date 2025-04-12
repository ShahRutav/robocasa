import yaml
import random
import os
import glob

# Define lists of replacement options
# 6, 10, 11, 0, 2, 3, 7, 5, 1
TRAIN_STYLES = [
    "INDUSTRIAL" "SCANDANAVIAN",
    "COASTAL",
    "MODERN_1",
    "TRADITIONAL_1",
    "TRADITIONAL_2",
    "FARMHOUSE",
    "TRANSITIONAL_1",
    "TRANSITIONAL_2",
]
TEST_STYLES = [
    "MODERN_2",
    "RUSTIC",
    "MEDITERRANEAN",
]

TRAIN_MICROWAVE_OPTIONS = [
    "hamilton_beach",
    "gray",
    "pack_1",
]
TEST_MICROWAVE_OPTIONS = ["standard"]
TRAIN_SINK_OPTIONS = [
    "1_bin_storage_right_dark",
    "2_bins_stainless",
    "concrete_sink",
    "white_sink",
]
TEST_SINK_OPTIONS = [
    "1_bin_wide_top_handle",
]
TRAIN_STOVE_OPTIONS = [
    "basic_sleek_induc",
    "frigidaire_gas",
]
TEST_STOVE_OPTIONS = [
    "square_gas",
]
"""
TRAIN_COUNTER_OPTIONS = [
    "blue_gray_base_white_top",
    "dark_blue_base_marble_top",
    "default",
    "gray_base_marble_top",
    "gray_wood_base_marble_top",
    "green_base_wood_top",
    "warm_wood_base_granite_top",
    "white_base_marble_top",
    "white_base_marble_top_2",
]
TEST_COUNTER_OPTIONS = [
    "white_base_marble_top_3",
    "gray_base_marble_top_2",
    "wood_base_dark_marble_top",
]
"""


def replace_fixtures(yaml_files, output_dir, mode="train", gen_n=0):
    """
    Reads a YAML file, replaces fixture values, and saves the modified file.

    Args:
        yaml_file_path: Path to the input YAML file
        output_dir: Directory to save the modified file (defaults to same directory as input)
    """
    yaml_file_path = random.choice(yaml_files)
    # Read the YAML file
    with open(yaml_file_path, "r") as file:
        data = yaml.safe_load(file)

    if (mode == "train") or (mode == "l1"):
        MICROWAVE_OPTIONS = TRAIN_MICROWAVE_OPTIONS
        SINK_OPTIONS = TRAIN_SINK_OPTIONS
        STOVE_OPTIONS = TRAIN_STOVE_OPTIONS
    elif mode == "l2":
        MICROWAVE_OPTIONS = TRAIN_MICROWAVE_OPTIONS + TEST_MICROWAVE_OPTIONS
        SINK_OPTIONS = TRAIN_SINK_OPTIONS + TEST_SINK_OPTIONS
        STOVE_OPTIONS = TRAIN_STOVE_OPTIONS + TEST_STOVE_OPTIONS
    else:
        MICROWAVE_OPTIONS = TEST_MICROWAVE_OPTIONS
        SINK_OPTIONS = TEST_SINK_OPTIONS
        STOVE_OPTIONS = TEST_STOVE_OPTIONS
    # Replace values if they exist in the YAML
    if "microwave" in data:
        data["microwave"] = random.choice(MICROWAVE_OPTIONS)

    if "sink" in data:
        data["sink"] = random.choice(SINK_OPTIONS)

    if "stove" in data:
        data["stove"] = random.choice(STOVE_OPTIONS)

    # Determine output path
    filename = f"{gen_n:03d}.yaml"
    output_path = os.path.join(output_dir, filename)

    # Write the modified YAML
    with open(output_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

    print(f"Modified YAML saved to: {output_path}")
    return output_path


def process_directory(directory_path, output_dir, mode="train", n_generates=40):
    """
    Process all YAML files in a directory.

    Args:
        directory_path: Path to directory containing YAML files
        output_dir: Directory to save modified files
    """
    mode_str = mode
    if mode == "trian":
        mode_str = "l1"
    output_dir = os.path.join(output_dir, mode_str)
    os.makedirs(output_dir, exist_ok=True)

    yaml_files = os.listdir(directory_path)
    print(yaml_files)
    yaml_files = [
        os.path.join(directory_path, f) for f in yaml_files if f.endswith(".yaml")
    ]
    VALID_STYLES = (
        TRAIN_STYLES
        if ((mode == "train") or (mode == "l2") or (mode == "l1"))
        else TEST_STYLES
    )
    yaml_files = [
        f for f in yaml_files if f.split("/")[-1].split(".")[0].upper() in VALID_STYLES
    ]
    print(f"Found {len(yaml_files)} YAML files in {directory_path}")
    print(yaml_files)

    if not yaml_files:
        print(f"No YAML files found in {directory_path}")
        return

    for n_gen in range(n_generates):
        replace_fixtures(yaml_files, output_dir, mode, n_gen)


# Example usage:
if __name__ == "__main__":
    mode = "l2"
    n_generates = 40
    # Process a single file
    # replace_fixtures("path/to/your/file.yaml", "output_directory")

    # Or process all YAML files in a directory
    # process_directory("path/to/your/directory", "output_directory")

    # Example with current directory
    process_directory(
        "robocasa/models/assets/scenes/kitchen_styles",
        "robocasa/models/assets/scenes/kitchen_styles",
        mode,
        n_generates,
    )
