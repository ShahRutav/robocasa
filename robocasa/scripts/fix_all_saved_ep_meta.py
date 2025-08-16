import os
import h5py
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from termcolor import colored
import deepdiff
from deepdiff import DeepDiff
from easydict import EasyDict
import robocasa
from icrt.util.casa_utils import (
    make_env,
    get_env_meta_from_dataset,
    get_env_args_from_dataset,
)


def get_default_hdf5_paths() -> List[str]:
    """Get default list of HDF5 files from generate_imgs_from_demos.py"""
    base_dir = os.environ.get("CASAPLAY_DATAROOT", None)
    assert (
        base_dir is not None
    ), "CASAPLAY_DATAROOT environment variable must be set to the base directory of the dataset."
    # Base paths for different tasks (same as in generate_imgs_from_demos.py)
    base_paths = [
        # L1 tasks
        f"{base_dir}/PlayEnvFinal/final_prompts/PnPSinkToRightCounterPlate/003",
        f"{base_dir}/PlayEnvFinal/final_prompts/PnPSinkToCabinet/003",
        f"{base_dir}/PlayEnvFinal/final_prompts/TurnOnFaucet/003",
        f"{base_dir}/PlayEnvFinal/final_prompts/CloseLeftCabinetDoor/003",
        # L2 tasks
        f"${base_dir}/home/rutavms/datasets/robocasa/datasets/PlayEnvFinal/final_prompts/PnPSinkToRightCounterPlateL2/003",
        f"${base_dir}/home/rutavms/datasets/robocasa/datasets/PlayEnvFinal/final_prompts/PnPSinkToCabinetL2/003",
        f"${base_dir}/home/rutavms/datasets/robocasa/datasets/PlayEnvFinal/final_prompts/CloseRightCabinetDoorL2/003",
        f"${base_dir}/home/rutavms/datasets/robocasa/datasets/PlayEnvFinal/final_prompts/CloseLeftCabinetDoorL2/003",
        # L3 tasks
        f"{base_dir}/PlayEnvFinal/final_prompts/CloseLeftCabinetDoorL3/003",
        f"{base_dir}/PlayEnvFinal/final_prompts/PnPSinkToRightCounterPlateL3/003",
        f"{base_dir}/PlayEnvFinal/final_prompts/PnPSinkToMicrowaveTopL3/003",
        f"{base_dir}/PlayEnvFinal/final_prompts/TurnOnFaucetL3/003",
    ]

    # Use demo.hdf5 files (contain states) for ep_meta comparison
    hdf5_paths = [f"{path}/demo.hdf5" for path in base_paths]

    # Filter out paths that don't exist
    existing_paths = [path for path in hdf5_paths if os.path.exists(path)]

    if not existing_paths:
        print(
            colored(
                "Warning: No default HDF5 files found. Please check the paths in get_default_hdf5_paths().",
                "yellow",
            )
        )
        return []

    print(colored(f"Found {len(existing_paths)} default HDF5 files", "green"))
    return existing_paths


def get_ep_metadata_from_dataset(dataset_path: str, key: str = None) -> Dict[str, Any]:
    """Extract episode metadata from HDF5 dataset."""
    with h5py.File(dataset_path, "r") as f:
        if "data" in f:
            data = f["data"]
        else:
            data = f
        if key is None:
            key = list(data.keys())[0]
        ep_meta = data[key].attrs["ep_meta"]
        if isinstance(ep_meta, str):
            ep_meta = json.loads(ep_meta)
    return ep_meta


def create_dummy_keys_info():
    """Create a dummy keys_info structure for environment creation."""
    return {
        "image_keys": [],
        "proprio_keys": [],
        "action_keys": ["actions"],
        "low_dim_keys": [],
        "bbox_keys": [],
    }


def create_dummy_args():
    """Create a dummy args object for environment creation."""

    class DummyArgs:
        def __init__(self):
            self.robots = "DemoTwoHand"
            self.controller = None
            self.render = False
            self.control_freq = 20
            self.reset_mode = None
            self.task_name = None

    return DummyArgs()


def process_hdf5_file(file_path: str) -> Dict[str, Any]:
    """Process a single HDF5 file and return metadata comparison."""
    print(colored(f"\nProcessing file: {file_path}", "cyan"))

    # Extract original ep_meta from the dataset
    original_ep_meta = get_ep_metadata_from_dataset(file_path)
    print(colored(f"Original ep_meta keys: {list(original_ep_meta.keys())}", "green"))

    # Create dummy keys_info and args for environment creation
    keys_info = create_dummy_keys_info()
    args = create_dummy_args()

    # Create the robocasa environment
    print(colored("Creating robocasa environment...", "yellow"))
    env, env_kwargs = make_env(file_path, keys_info, args)

    # Set the original ep_meta
    print(colored("Setting original ep_meta...", "yellow"))
    env.set_ep_meta(original_ep_meta)

    # Retrieve the new ep_meta using get_ep_meta
    print(colored("Retrieving new ep_meta...", "yellow"))
    new_ep_meta = env.get_ep_meta()

    # Compare the differences
    print(colored("Comparing ep_meta differences...", "yellow"))
    diff = DeepDiff(original_ep_meta, new_ep_meta, ignore_order=True)

    # Clean up environment
    if hasattr(env, "_destroy_viewer"):
        env._destroy_viewer()

    return {
        "file_path": file_path,
        "original_ep_meta": original_ep_meta,
        "new_ep_meta": new_ep_meta,
        "differences": diff.to_dict() if diff else {},
        "has_differences": bool(diff),
        "env_name": env_kwargs.get("env_name", "Unknown"),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare ep_meta between original and retrieved from robocasa environment"
    )
    parser.add_argument(
        "--hdf5_files",
        nargs="*",
        default=None,
        help="List of HDF5 files to process (if None, uses default list)",
    )
    parser.add_argument(
        "--output_file", type=str, default=None, help="Output JSON file to save results"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )
    parser.add_argument(
        "--differences_list_file",
        type=str,
        default="ep_meta_differences.txt",
        help="File to store list of paths with differences",
    )

    args = parser.parse_args()

    # If no HDF5 files provided, use default list
    if args.hdf5_files is None or len(args.hdf5_files) == 0:
        print(colored("No HDF5 files specified, using default list...", "yellow"))
        hdf5_files = get_default_hdf5_paths()
        if not hdf5_files:
            print(
                colored(
                    "Error: No default HDF5 files found. Please specify files manually with --hdf5_files.",
                    "red",
                )
            )
            return
    else:
        hdf5_files = args.hdf5_files

    print(colored(f"Processing {len(hdf5_files)} HDF5 files", "blue"))

    results = []
    total_files = len(hdf5_files)
    files_with_differences = 0

    for i, file_path in enumerate(hdf5_files):
        print(
            colored(
                f"\n[{i+1}/{total_files}] Processing: {Path(file_path).name}", "blue"
            )
        )

        result = process_hdf5_file(file_path)
        results.append(result)

        if result.get("has_differences", False):
            files_with_differences += 1
            print(colored("✓ Differences found", "red"))
        else:
            print(colored("✓ No differences", "green"))

    # Summary
    print(colored(f"\n{'='*60}", "blue"))
    print(colored("SUMMARY", "blue"))
    print(colored(f"{'='*60}", "blue"))
    print(colored(f"Total files processed: {total_files}", "blue"))
    print(
        colored(
            f"Files with differences: {files_with_differences}",
            "red" if files_with_differences > 0 else "green",
        )
    )
    print(
        colored(
            f"Files without differences: {total_files - files_with_differences}",
            "green",
        )
    )

    # Print detailed results for files with differences
    if files_with_differences > 0:
        print(colored(f"\n{'='*60}", "red"))
        print(colored("FILES WITH DIFFERENCES:", "red"))
        print(colored(f"{'='*60}", "red"))

        for result in results:
            if result.get("has_differences", False):
                print(colored(f"\nFile: {Path(result['file_path']).name}", "red"))
                print(
                    colored(
                        f"Environment: {result.get('env_name', 'Unknown')}", "yellow"
                    )
                )

                # Print detailed differences
                print(colored("\n--- DETAILED DIFFERENCES ---", "cyan"))
                diff = result["differences"]

                try:
                    if "values_changed" in diff:
                        print(colored("\nValues Changed:", "yellow"))
                        for change in diff["values_changed"].items():
                            path = change[0]
                            old_value = change[1]["old_value"]
                            new_value = change[1]["new_value"]
                            print(f"  Path: {path}")
                            print(f"    Old: {old_value}")
                            print(f"    New: {new_value}")

                    if "dictionary_item_added" in diff:
                        print(colored("\nItems Added:", "green"))
                        for item in diff["dictionary_item_added"]:
                            print(f"  Added: {item}")

                    if "dictionary_item_removed" in diff:
                        print(colored("\nItems Removed:", "red"))
                        for item in diff["dictionary_item_removed"]:
                            print(f"  Removed: {item}")

                    if "type_changes" in diff:
                        print(colored("\nType Changes:", "magenta"))
                        for change in diff["type_changes"].items():
                            path = change[0]
                            old_type = change[1]["old_type"]
                            new_type = change[1]["new_type"]
                            print(f"  Path: {path}")
                            print(f"    Old Type: {old_type}")
                            print(f"    New Type: {new_type}")

                    if "iterable_item_added" in diff:
                        print(colored("\nIterable Items Added:", "green"))
                        for change in diff["iterable_item_added"].items():
                            path = change[0]
                            value = change[1]
                            print(f"  Path: {path}")
                            print(f"    Added: {value}")

                    if "iterable_item_removed" in diff:
                        print(colored("\nIterable Items Removed:", "red"))
                        for change in diff["iterable_item_removed"].items():
                            path = change[0]
                            value = change[1]
                            print(f"  Removed: {value}")

                except Exception as e:
                    print(
                        colored(
                            f"\nWarning: Error processing differences: {str(e)}",
                            "yellow",
                        )
                    )
                    print(colored("Showing raw differences instead:", "yellow"))
                    print(json.dumps(diff, indent=2, default=str))

                # Print full ep_meta comparison if verbose mode is enabled
                if args.verbose:
                    print(colored("\n--- FULL EP_META COMPARISON ---", "cyan"))
                    try:
                        print("Original ep_meta:")
                        print(
                            json.dumps(
                                result["original_ep_meta"], indent=2, default=str
                            )
                        )
                        print("\nNew ep_meta:")
                        print(json.dumps(result["new_ep_meta"], indent=2, default=str))
                    except Exception as e:
                        print(
                            colored(
                                f"Warning: Could not serialize ep_meta: {str(e)}",
                                "yellow",
                            )
                        )
                        print("Original ep_meta:")
                        print(str(result["original_ep_meta"]))
                        print("\nNew ep_meta:")
                        print(str(result["new_ep_meta"]))

                print(colored("-" * 60, "cyan"))

    # Save results to JSON file if requested
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert DeepDiff objects to serializable format and handle non-serializable objects
        serializable_results = []
        for result in results:
            serializable_result = result.copy()

            # Handle DeepDiff objects
            if "differences" in serializable_result:
                serializable_result["differences"] = serializable_result["differences"]

            # Handle ep_meta objects - convert to string representation if not serializable
            if "original_ep_meta" in serializable_result:
                try:
                    json.dumps(serializable_result["original_ep_meta"])
                except (TypeError, ValueError):
                    serializable_result["original_ep_meta"] = str(
                        serializable_result["original_ep_meta"]
                    )

            if "new_ep_meta" in serializable_result:
                try:
                    json.dumps(serializable_result["new_ep_meta"])
                except (TypeError, ValueError):
                    serializable_result["new_ep_meta"] = str(
                        serializable_result["new_ep_meta"]
                    )

            serializable_results.append(serializable_result)

        try:
            with open(output_path, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)
            print(colored(f"\nResults saved to: {output_path}", "green"))
        except Exception as e:
            print(
                colored(
                    f"\nWarning: Could not save results to JSON file: {str(e)}",
                    "yellow",
                )
            )
            print(colored("Results will still be displayed in terminal", "yellow"))

    # Create and save list of files with differences
    if files_with_differences > 0:
        differences_list_path = Path(args.differences_list_file)
        differences_list_path.parent.mkdir(parents=True, exist_ok=True)

        with open(differences_list_path, "w") as f:
            f.write(f"# List of HDF5 files with ep_meta differences\n")
            f.write(f"# Total files processed: {total_files}\n")
            f.write(f"# Files with differences: {files_with_differences}\n")
            f.write(
                f"# Files without differences: {total_files - files_with_differences}\n"
            )
            f.write(f"# Generated on: {os.popen('date').read().strip()}\n\n")

            for result in results:
                if result.get("has_differences", False):
                    f.write(f"{result['file_path']}\n")

        print(
            colored(
                f"\nList of files with differences saved to: {differences_list_path}",
                "green",
            )
        )

        # Also print the list at the end
        print(colored(f"\n{'='*60}", "red"))
        print(colored("LIST OF FILES WITH DIFFERENCES:", "red"))
        print(colored(f"{'='*60}", "red"))
        for result in results:
            if result.get("has_differences", False):
                print(colored(f"{result['file_path']}", "red"))

    # Return summary for potential script usage
    return {
        "total_files": total_files,
        "files_with_differences": files_with_differences,
        "files_without_differences": total_files - files_with_differences,
        "results": results,
    }


if __name__ == "__main__":
    main()
