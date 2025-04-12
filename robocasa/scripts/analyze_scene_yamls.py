import os
import argparse
import yaml

# pretty print
import pprint

from robocasa.models.scenes.scene_registry import StyleType

### floor, wall, counter is unique across all styles


def main(args):
    scene_dir = "robocasa/models/assets/scenes/kitchen_styles/"
    yaml_files = os.listdir(scene_dir)
    yaml_files = [f for f in yaml_files if f.endswith(".yaml")]
    style_data = {}
    for yaml_file in yaml_files:
        style_name = yaml_file.split(".")[0].upper()
        if style_name == "PLAYGROUND":
            continue
        style_id = StyleType[str(style_name)].value
        print(f"{style_id}: {yaml_file}")
        file_path = os.path.join(scene_dir, yaml_file)
        assert os.path.exists(file_path), f"File not found: {file_path}"
        with open(file_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        style_data[style_id] = {}
        style_data[style_id]["file"] = yaml_file
        style_data[style_id]["style_name"] = style_name
        style_data[style_id]["sink"] = data["sink"]
        style_data[style_id]["stove"] = data["stove"]
        style_data[style_id]["microwave"] = data["microwave"]
        style_data[style_id]["counter"] = data["counter"]
        style_data[style_id]["floor"] = data["floor"]
        style_data[style_id]["wall"] = data["wall"]
    # print(style_data)

    microwave_data = {}
    for style_id in style_data:
        m_name = style_data[style_id]["microwave"]
        if m_name not in microwave_data:
            microwave_data[m_name] = []
        microwave_data[m_name].append(style_id)
    print("Microwaves:")
    pprint.pprint(microwave_data)

    sink_data = {}
    for style_id in style_data:
        s_name = style_data[style_id]["sink"]
        if s_name not in sink_data:
            sink_data[s_name] = []
        sink_data[s_name].append(style_id)
    print("Sinks:")
    pprint.pprint(sink_data)

    stove_data = {}
    for style_id in style_data:
        s_name = style_data[style_id]["stove"]
        if s_name not in stove_data:
            stove_data[s_name] = []
        stove_data[s_name].append(style_id)
    print("Stoves:")
    pprint.pprint(stove_data)

    counter_data = {}
    for style_id in style_data:
        c_name = style_data[style_id]["counter"]["default"]
        if c_name not in counter_data:
            counter_data[c_name] = []
        counter_data[c_name].append(style_id)
    print("Counters:")
    pprint.pprint(counter_data)

    floor_data = {}
    for style_id in style_data:
        f_name = style_data[style_id]["floor"]
        if f_name not in floor_data:
            floor_data[f_name] = []
        floor_data[f_name].append(style_id)
    print("Floors:")
    pprint.pprint(floor_data)

    wall_data = {}
    for style_id in style_data:
        w_name = style_data[style_id]["wall"]
        if w_name not in wall_data:
            wall_data[w_name] = []
        wall_data[w_name].append(style_id)
    print("Walls:")
    pprint.pprint(wall_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str, default="temp")
    parser.add_argument("--environment", type=str, default="SinkPlayEnv")
    parser.add_argument("--seed", "-s", type=int, default=0)
    args = parser.parse_args()
    main(args)
