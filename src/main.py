import json
from pathlib import Path

import pandas as pd
from detquantlib.outputs import OutputSet


def main(settings_dict=None):
    if settings_dict is None:
        # Load settings from input json file
        input_dir = Path.cwd().joinpath("Inputs")
        settings_json_dir = input_dir.joinpath("settings.json")
        with open(settings_json_dir) as f:
            settings_dict = json.load(f)

    df = pd.DataFrame(index=[0], columns=["total"])
    df["total"] = settings_dict["a"] - settings_dict["b"]

    output_set = OutputSet()
    output_set.add_item(data=df, filename="df", extension="csv")

    if settings_dict["create_output_files"]:
        for item in output_set.output_items:
            item.export_to_file()

    return output_set
