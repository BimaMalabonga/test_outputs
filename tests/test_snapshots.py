# Python built-in packages
import json
import os
import shutil
from pathlib import Path
from typing import Literal

# Third-party packages
import numpy as np
import pandas as pd
from detquantlib.outputs import OutputSet
from dotenv import load_dotenv

# Internal modules
from src.main import main

# Constants
TESTS_BASE_DIR = Path(__file__).parent
PROJECT_BASE_DIR = TESTS_BASE_DIR.parent
SNAPSHOTS_DIR = TESTS_BASE_DIR.joinpath("snapshots")
EXPECTED_OUTPUTS_FOLDER = "ExpectedOutputs"
OUTPUTS_FOLDER_NAME = "TempOutputs"
INPUTS_FOLDER_NAME = "Inputs"
SETTINGS_FILENAME = "settings.json"
ABSOLUTE_TOLERANCE = 1e-8
RELATIVE_TOLERANCE = 0

# Load environment variables from local file '.env' (if file exists)
load_dotenv()


def test_snapshots(run_type: Literal["compare", "update", "create"] = "compare"):
    """
    Main function to run snapshot tests. The function can fulfill one of three purposes:
        - Standard snapshot test run. The function then goes through the following steps:
            1. Find all existing test cases
            2. Run each test case
            3. Compare the actual model outputs with the expected outputs
        - Update the expected outputs of all test cases
        - Create a new test case, using the settings currently stored in the project's input folder

    Args:
        run_type: Determines the function's run type. Available options:
            - run_type="compare": Standard snapshot test run, comparing actual model outputs
                with expected outputs
            - run_type="update": Updates the expected outputs of all test cases
            - run_type="create": Creates a new test case, using the settings currently stored in
                the project's input folder

    Raises:
        ValueError: Raises an error if the value of the input argument 'run_type' is invalid.
    """
    # Input validation
    accepted_run_types = ["compare", "update", "create"]
    if run_type not in accepted_run_types:
        accepted_run_types_str = ", ".join(f"'{item}'" for item in accepted_run_types)
        raise ValueError(
            f"Invalid input 'run_type'='{run_type}'. Accepted values: {accepted_run_types_str}."
        )

    # Find test cases
    if run_type == "create":
        cases = [create_new_case_folder()]
    else:
        cases = find_cases()

    for case_dir in cases:
        # Load settings
        inputs_base_dir = PROJECT_BASE_DIR if run_type == "create" else case_dir
        settings = load_settings(base_dir=inputs_base_dir)

        # Run the model
        output_set = main(settings)

        # Define output folders
        expected_outputs_base_dir = case_dir.joinpath(EXPECTED_OUTPUTS_FOLDER)
        outputs_base_dir = case_dir.joinpath(OUTPUTS_FOLDER_NAME)

        if run_type == "create":
            # Copy input settings and export outputs
            export_settings(case_dir=case_dir, settings=settings)
            export_outputs(outputs_dir=expected_outputs_base_dir, output_set=output_set)
        if run_type == "update":
            # Export outputs
            export_outputs(outputs_dir=expected_outputs_base_dir, output_set=output_set)
        elif run_type == "compare":
            # Temporarily export outputs
            export_outputs(outputs_dir=outputs_base_dir, output_set=output_set)
            # Compare with expected outputs
            compare_outputs(
                expected_outputs_base_dir=expected_outputs_base_dir,
                outputs_base_dir=outputs_base_dir,
            )
            # Remove temporary outputs
            shutil.rmtree(outputs_base_dir)


def find_cases() -> list:
    """
    Finds all existing snapshot test cases.

    Returns:
        List of test case directories
    """
    cases = list()
    if SNAPSHOTS_DIR.is_dir():
        cases = [f for f in SNAPSHOTS_DIR.iterdir() if f.is_dir() and f.name.startswith("Case")]
    return cases


def create_new_case_folder() -> Path:
    """
    Creates the folder that will contain a new test case.

    Returns:
        Directory of the new test case
    """
    # Find all existing cases
    cases = find_cases()
    case_numbers = [int(c.name[4:]) for c in cases]

    # Determine new case number
    new_case_number = None
    for i in range(1, max(case_numbers)):
        if i not in case_numbers:
            new_case_number = i
    if new_case_number is None:
        new_case_number = max(case_numbers) + 1

    # Create new case folder
    new_case = f"Case{new_case_number:02d}"
    new_case_dir = SNAPSHOTS_DIR.joinpath(new_case)
    new_case_dir.mkdir(parents=True, exist_ok=True)

    return new_case_dir


def load_settings(base_dir: Path) -> dict:
    """
    Loads model settings.

    Args:
        base_dir: Base directory containing the model's input folder

    Returns:
        Model settings
    """
    settings_dir = base_dir.joinpath(INPUTS_FOLDER_NAME, SETTINGS_FILENAME)
    with open(settings_dir) as f:
        settings = json.load(f)
    settings["create_output_files"] = False
    return settings


def export_settings(case_dir: Path, settings: dict):
    """
    Exports the model settings to a json file.

    Args:
        case_dir: Directory of the test case in which model settings should be exported
        settings: Model settings
    """
    # Create inputs folder (if it already exists, we first delete it to clear its content)
    inputs_dir = case_dir.joinpath(INPUTS_FOLDER_NAME)
    if inputs_dir.is_dir():
        shutil.rmtree(inputs_dir)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Export model settings to inputs folder
    settings_json_dir = inputs_dir.joinpath(SETTINGS_FILENAME)
    with open(settings_json_dir, "w") as f:
        json.dump(settings, f, indent=4)


def export_outputs(outputs_dir: Path, output_set: OutputSet):
    """
    Exports the model outputs.

    Args:
        outputs_dir: Output folder directory
        output_set: Model outputs
    """
    # Create outputs folder (if it already exists, we first delete it to clear its content)
    if outputs_dir.is_dir():
        shutil.rmtree(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Export model outputs
    # Note: We do not store html files, because they contain metadata values that change every
    # time (such as unique identifiers).
    for item in output_set.output_items:
        if item.extension != "html":
            item.export_to_file(folder_dir=outputs_dir)


def compare_outputs(expected_outputs_base_dir: Path, outputs_base_dir: Path):
    """
    Compare the model outputs with the expected outputs, and flags differences.

    Args:
        expected_outputs_base_dir: Expected outputs folder directory
        outputs_base_dir: Model outputs folder directory
    """
    # Loop over files and sub-folders located the expected outputs folder
    for base_dir, folder_names, filenames in os.walk(expected_outputs_base_dir):
        for f in filenames:
            # Define path in expected outputs folder
            expected_output_file_dir = Path(base_dir).joinpath(f)

            # Define path in model outputs folder
            dir_suffix = expected_output_file_dir.relative_to(expected_outputs_base_dir)
            output_file_dir = outputs_base_dir.joinpath(dir_suffix)

            # Assert output file
            assert_file(output_file_dir, expected_output_file_dir)


def assert_file(output_file_dir: Path, expected_output_file_dir: Path):
    """
    Entry point function to assert that an output file is equal to the corresponding expected
    output file. Depending on the file extension, the procedure to open and assert the files
    will differ.

    Args:
        output_file_dir: Output file directory
        expected_output_file_dir: Expected output file directory

    Raises:
        ValueError: Raises an error if the file extension is not supported
    """
    extension = expected_output_file_dir.suffix
    if extension == ".csv":
        assert_csv(output_file_dir, expected_output_file_dir)
    elif extension == ".json":
        assert_json(output_file_dir, expected_output_file_dir)
    elif extension == ".html":
        # We do not store html files, because they contain metadata values that change every
        # time (such as unique identifiers).
        pass
    elif extension == ".npz":
        assert_npz(output_file_dir, expected_output_file_dir)
    else:
        raise ValueError(f"File extension '{extension}' is not supported.")


def assert_csv(output_file_dir: Path, expected_output_file_dir: Path):
    """
    Imports an output csv file and asserts that it is equal to the corresponding expected
    output csv file.

    Args:
        output_file_dir: Output file directory
        expected_output_file_dir: Expected output file directory
    """
    data = pd.read_csv(output_file_dir)
    expected_data = pd.read_csv(expected_output_file_dir)
    pd.testing.assert_frame_equal(
        data, expected_data, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
    )


def assert_json(output_file_dir: Path, expected_output_file_dir: Path):
    """
    Imports an output json file and asserts that it is equal to the corresponding expected
    output json file.

    Args:
        output_file_dir: Output file directory
        expected_output_file_dir: Expected output file directory
    """
    with open(output_file_dir) as f:
        data = json.load(f)
    with open(expected_output_file_dir) as f:
        expected_data = json.load(f)
    assert data == expected_data


def assert_npz(output_file_dir: Path, expected_output_file_dir: Path):
    """
    Imports an output npz file and asserts that it is equal to the corresponding expected
    output npz file.

    Args:
        output_file_dir: Output file directory
        expected_output_file_dir: Expected output file directory

    Raises:
        KeyError: Raises an error when the keys of the npz archive files do not match.
    """
    data = np.load(output_file_dir)
    expected_data = np.load(expected_output_file_dir)
    if set(data.files) != set(expected_data.files):
        raise KeyError("The .npz archives have different keys.")
    else:
        # Compare all files in archives
        for name in expected_data.files:
            np.testing.assert_allclose(
                data[name], expected_data[name], atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE
            )
