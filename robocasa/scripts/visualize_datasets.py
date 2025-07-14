import os
import random
import json
import numpy as np
import h5py
import argparse
import plotly.graph_objects as go
import plotly.io as pio
import robocasa

pio.renderers.default = "browser"


def plotly_draw_3d_pcd(
    pcd_points,
    pcd_colors=None,
    addition_points=None,
    marker_size=3,
    equal_axis=True,
    title="",
    offline=False,
    no_background=False,
    default_rgb_str="(255,0,0)",
    additional_point_draw_lines=False,
    uniform_color=False,
):

    if pcd_colors is None:
        color_str = [f"rgb{default_rgb_str}" for _ in range(pcd_points.shape[0])]
    else:
        color_str = [
            "rgb(" + str(r) + "," + str(g) + "," + str(b) + ")"
            for r, g, b in pcd_colors
        ]

    # Extract x, y, and z columns from the point cloud
    x_vals = pcd_points[:, 0]
    y_vals = pcd_points[:, 1]
    z_vals = pcd_points[:, 2]

    # Create the scatter3d plot
    rgbd_scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode="markers",
        marker=dict(size=3, color=color_str, opacity=0.8),
    )
    data = [rgbd_scatter]
    if addition_points is not None:
        assert addition_points.shape[-1] == 3
        # check if addition_points are three dimensional
        if len(addition_points.shape) == 2:
            addition_points = [addition_points]
        for points in addition_points:
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            if additional_point_draw_lines:
                mode = "lines+markers"
            else:
                mode = "markers"
            marker_dict = dict(size=marker_size, opacity=0.8)

            if uniform_color:
                marker_dict["color"] = f"rgb{default_rgb_str}"
            rgbd_scatter2 = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode=mode,
                marker=marker_dict,
            )
            data.append(rgbd_scatter2)

    if equal_axis:
        scene_dict = dict(
            aspectmode="data",
        )
    else:
        scene_dict = dict()
    # Set the layout for the plot
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        # axes range
        scene=scene_dict,
        title=dict(text=title, automargin=True),
    )

    fig = go.Figure(data=data, layout=layout)

    if no_background:
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showbackground=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    showaxeslabels=False,
                    visible=False,
                ),
                yaxis=dict(
                    showbackground=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    showaxeslabels=False,
                    visible=False,
                ),
                zaxis=dict(
                    showbackground=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    showaxeslabels=False,
                    visible=False,
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
            plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
            margin=dict(l=0, r=0, b=0, t=0),  # No margins
            showlegend=False,
        )

    if not offline:
        fig.show()
    else:
        return fig


if __name__ == "__main__":
    # Load every dataset in the dataset folder; passed will be an argument with list of the dataset hdf5 files
    # Our goal is to actions [x,y,z,ax,ay,az,gripper,base_x, base_y, vel_x, vel_y, control_mode] visualize the dataset using the function plotly_draw_3d_pcd
    # we want to visualize the xyz in one plot; ax,ay,az in another plot; base_x, base_y in another plot; vel_x, vel_y in another plot; control_mode and gripper mean, min, and max values
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_json", type=str, default="robocasa/configs/dataset_example.json"
    )
    args = parser.parse_args()

    with open(args.dataset_json, "r") as f:
        dataset_json = json.load(f)

    dataset_path = dataset_json["dataset_path"]
    xyz_actions_list = []
    axayaz_actions_list = []
    base_xy_actions_list = []
    vel_xy_actions_list = []
    control_mode_actions_list = []
    gripper_actions_list = []
    labels_list = []
    # generate colors equivalent to the number of datasets with random colors
    colors_list = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(len(dataset_path))
    ]

    for ind, rel_file_path in enumerate(dataset_path):
        robocasa_path = robocasa.__path__[0].split("/")[:-1]
        path = "/".join(robocasa_path + rel_file_path.split("/"))
        with h5py.File(path, "r") as f:
            data = f["data"]
            key = list(data.keys())[0]
            actions = data[key]["actions"][:]
            print(actions.shape)
            # print(actions[0])
            print(data[key]["actions"][0])
            xyz_actions = actions[:, :3]
            axayaz_actions = actions[:, 3:6]
            gripper_actions = actions[:, 6]
            base_xy_actions = actions[:, 7:9]
            vel_xy_actions = actions[:, 9:11]
            control_mode_actions = actions[:, 11]
            label_name = path.split("/")[-3]
            labels_list.append(label_name)
            print(f"label_name: {label_name}")
            xyz_actions_list.append(xyz_actions)
            axayaz_actions_list.append(axayaz_actions)
            gripper_actions_list.append(gripper_actions)
            base_xy_actions_list.append(base_xy_actions)
            vel_xy_actions_list.append(vel_xy_actions)
            control_mode_actions_list.append(control_mode_actions)
    xyz_actions_np = np.concatenate(xyz_actions_list, axis=0)
    # repeat the color for the each point
    colors_list = [
        colors_list[ind]
        for ind in range(len(xyz_actions_list))
        for _ in range(xyz_actions_list[ind].shape[0])
    ]
    plotly_draw_3d_pcd(xyz_actions_np, title="xyz_actions", pcd_colors=colors_list)
    plotly_draw_3d_pcd(xyz_actions_np, title="xyz_actions", pcd_colors=colors_list)
    # plotly_draw_3d_pcd(axayaz_actions_np, title="axayaz_actions", pcd_colors=colors_list)
    # plotly_draw_3d_pcd(base_xy_actions_np, title="base_xy_actions", pcd_colors=colors_list)
