import io
import logging
import os
import re
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import h5py

from . import data_utils
from . import session


def display_plot_history_sidebar():
    st.sidebar.header("Previously Generated Plots")
    if not st.session_state.get("plots_history"):
        st.session_state["plots_history"] = []
    if st.session_state["plots_history"]:
        for idx, plot_info in enumerate(st.session_state["plots_history"]):
            st.sidebar.write(f"**{idx + 1}. {plot_info['plot_name']}**")
            st.sidebar.pyplot(plot_info["figure"])
    else:
        st.sidebar.write("No plots have been generated yet.")


def display_structure(structure, file_name, file_path, parent_path="", indent_level=0):
    logging.debug(f"Displaying structure for file: {file_name} | Path: {parent_path}")
    for key, value in structure.items():
        if key.startswith("_"):
            continue
        current_path = f"{parent_path}/{key}" if parent_path else key
        if isinstance(value, dict):
            if value.get("_type") == "Dataset":
                button_key = f"{file_name}_{current_path}"
                if indent_level == 0:
                    if st.button(f"{key}", key=button_key):
                        st.session_state["selected_dataset"] = (
                            file_name,
                            file_path,
                            current_path,
                        )
                else:
                    indentation_width = min(indent_level * 0.05, 0.5)
                    remaining_width = 1.0 - indentation_width
                    columns = st.columns([indentation_width, remaining_width])
                    with columns[1]:
                        if st.button(f"{key}", key=button_key):
                            st.session_state["selected_dataset"] = (
                                file_name,
                                file_path,
                                current_path,
                            )
            else:
                if indent_level == 0:
                    with st.expander(f"{key}", expanded=False):
                        display_structure(
                            value,
                            file_name,
                            file_path,
                            current_path,
                            indent_level + 1,
                        )
                else:
                    indentation_width = min(indent_level * 0.05, 0.5)
                    remaining_width = 1.0 - indentation_width
                    columns = st.columns([indentation_width, remaining_width])
                    with columns[1]:
                        with st.expander(f"{key}", expanded=False):
                            display_structure(
                                value,
                                file_name,
                                file_path,
                                current_path,
                                indent_level + 1,
                            )


def display_dataset_content(file_name, file_path, dataset_path):
    data = data_utils.load_dataset(file_path, dataset_path)
    if data is None:
        return
    st.write(f"### Dataset: `{dataset_path}` in `{file_name}`")
    if isinstance(data, np.ndarray):
        if data.dtype.names:
            df = pd.DataFrame(data).applymap(data_utils.decode_if_bytes)
            if dataset_path == "Data/Channel names":
                st.write("#### Select Channels by Checking the Boxes Below")
                df = df.reset_index().rename(columns={"index": "Row Number"})
                selected_rows = []
                for i, row in df.iterrows():
                    base = row[df.columns[1]]
                    extra = row[df.columns[2]]
                    label = f"{row['Row Number']}: {base}" + (f" ({extra})" if extra else "")
                    if st.checkbox(label, key=f"select_channel_{i}"):
                        selected_rows.append(row)

                if selected_rows:
                    if st.button("Add Selected Channels", key="add_selected_channels"):
                        channel_info = [
                            {
                                "file_name": file_name,
                                "channel_name": row[df.columns[1]],
                                "file_path": file_path,
                                "data_path": "Data/Data",
                                "channel_index": row["Row Number"],
                            }
                            for row in selected_rows
                        ]
                        session.add_selected_channels(channel_info)
                        st.success(f"Added {len(channel_info)} channels from `{file_name}`.")
                else:
                    st.info("Check one or more boxes above to select channels.")

                st.write("#### Select Channels by Selecting Rows Below")
                df.reset_index(inplace=True)
                df.rename(columns={"index": "Row Number"}, inplace=True)
                st.dataframe(df)
                row_numbers = df["Row Number"].tolist()
                selected_rows = st.multiselect(
                    f"Select Channel Rows from `{file_name}` (by Row Number)",
                    options=row_numbers,
                    key="selected_channel_rows",
                )
                if selected_rows:
                    selected_channels = df[df["Row Number"].isin(selected_rows)]
                    channel_info = [
                        {
                            "file_name": file_name,
                            "channel_name": row[df.columns[1]],
                            "file_path": file_path,
                            "data_path": "Data/Data",
                            "channel_index": row["Row Number"],
                        }
                        for _, row in selected_channels.iterrows()
                    ]
                    if st.button("Add Selected Channels", key="add_selected_channels"):
                        session.add_selected_channels(channel_info)
                        st.success(f"Added {len(channel_info)} channels from `{file_name}`.")
                else:
                    st.info("Select one or more channels from the list above to add to your selection.")
            else:
                st.dataframe(df.applymap(data_utils.format_numbers))
        else:
            if data.dtype.kind in {"i", "f"}:
                if data.ndim == 3 and data.shape[2] == 1:
                    data = data.squeeze(axis=2)
                display_data = data[:100, ...] if data.shape[0] > 100 else data
                df = pd.DataFrame(display_data)
                st.dataframe(df.applymap(data_utils.format_numbers))
                if data.ndim == 1 or data.shape[1] == 1:
                    fig, ax = plt.subplots()
                    ax.plot(data.flatten())
                    ax.set_title(f"Line Plot of {dataset_path}")
                    st.pyplot(fig)
                elif data.ndim == 2:
                    fig, ax = plt.subplots()
                    cax = ax.imshow(data, aspect="auto", cmap="viridis")
                    fig.colorbar(cax)
                    ax.set_title(f"Heatmap of {dataset_path}")
                    st.pyplot(fig)
                else:
                    st.write("Data format not supported for plotting.")
            elif data.dtype.kind in {"S", "U"}:
                if data.ndim == 0:
                    st.text(data_utils.decode_if_bytes(data))
                else:
                    st.write([data_utils.decode_if_bytes(x) for x in data.flatten()])
            else:
                st.write("Unsupported data type for preview.")
    else:
        st.write("The selected dataset is not a numpy array.")


def display_selected_channels():
    if not st.session_state["selected_channels"]:
        st.info("No channels have been selected yet.")
        return
    st.sidebar.header("Selected Channels")
    for idx, channel in enumerate(st.session_state["selected_channels"], start=1):
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(f"{idx}. **{channel['file_name']} - {channel['channel_name']}**")
        if col2.button("Remove", key=f"remove_{idx}"):
            session.remove_selected_channel(idx - 1)
            st.experimental_rerun()


def display_combined_data():
    if not st.session_state["selected_channels"]:
        return
    st.write("## Combined Data of Selected Channels")
    data_frames = []
    for channel in st.session_state["selected_channels"]:
        try:
            data = data_utils.load_dataset(channel["file_path"], channel["data_path"])
            if data is None:
                continue
            if data.ndim == 3 and data.shape[2] == 1:
                data = data.squeeze(axis=2)
            if data.ndim == 1:
                df_data = pd.DataFrame(
                    data, columns=[f"{channel['file_name']} - {channel['channel_name']}"]
                )
                data_frames.append(df_data)
            else:
                df_data = pd.DataFrame(data)
                col_idx = channel["channel_index"]
                if col_idx >= df_data.shape[1]:
                    st.error(
                        f"Channel index {col_idx} out of bounds for file `{channel['file_name']}`."
                    )
                    continue
                df_data = df_data.iloc[:, [col_idx]]
                df_data.columns = [f"{channel['file_name']} - {channel['channel_name']}"]
                data_frames.append(df_data)
        except Exception as e:
            st.error(
                f"Error loading data for channel `{channel['channel_name']}` from `{channel['file_name']}`: {e}"
            )
    if data_frames:
        max_length = max(df.shape[0] for df in data_frames)
        padded_data_frames = [df.reindex(range(max_length)) for df in data_frames]
        combined_data = pd.concat(padded_data_frames, axis=1)
        st.dataframe(combined_data.applymap(data_utils.format_numbers))
    else:
        st.error("No data available to display.")


def plot_selected_channels():
    if not st.session_state.get("selected_channels"):
        st.info("No channels have been selected to plot.")
        return
    st.write("## Generate Plot")
    x_axis_options = [
        f"{idx + 1}. {ch['file_name']} - {ch['channel_name']}"
        for idx, ch in enumerate(st.session_state["selected_channels"])
    ]
    x_axis_choice = st.selectbox("X-Axis Channel", options=x_axis_options, key="x_axis_choice")
    try:
        x_idx = x_axis_options.index(x_axis_choice)
    except ValueError:
        st.error("Invalid X-axis channel choice.")
        return
    x_channel = st.session_state["selected_channels"][x_idx]
    x_data = data_utils.load_dataset(x_channel["file_path"], x_channel["data_path"])
    if x_data is None:
        return
    if x_data.ndim == 3 and x_data.shape[2] == 1:
        x_data = x_data.squeeze(axis=2)
    if x_data.ndim == 1:
        x_values = x_data
    else:
        if x_channel["channel_index"] >= x_data.shape[1]:
            st.error("X-axis channel index out of bounds.")
            return
        x_values = x_data[:, x_channel["channel_index"]]
    y_axis_options = x_axis_options.copy()
    y_axis_choices = st.multiselect("Y-Axis Channels", options=y_axis_options, key="y_axis_choices")
    if not y_axis_choices:
        st.info("Select at least one Y-axis channel to proceed.")
        return
    right_axis_choices = st.multiselect("Channels on Right Axis", options=y_axis_options, key="right_axis_choices")
    left_axis_choices = [c for c in y_axis_choices if c not in right_axis_choices]
    if not left_axis_choices and not right_axis_choices:
        st.error("At least one Y-axis channel must remain on the left or right axis.")
        return
    with st.expander("Rename Y-Axis Channels", expanded=False):
        renamed_channels = {}
        for c in y_axis_choices:
            idx_c = y_axis_options.index(c)
            ch_info = st.session_state["selected_channels"][idx_c]
            original_name = f"{ch_info['file_name']} - {ch_info['channel_name']}"
            current_renamed = ch_info.get("renamed_name", ch_info["channel_name"])
            renamed_channels[c] = st.text_input(
                f"Rename Y-Channel '{original_name}'", value=current_renamed, key=f"rename_{idx_c}"
            )
        for c, new_name in renamed_channels.items():
            idx_c = y_axis_options.index(c)
            st.session_state["selected_channels"][idx_c]["renamed_name"] = new_name
    with st.expander("Naming (including legends)", expanded=False):
        show_legend = st.checkbox(
            "Show Legend", value=st.session_state.get("show_legend", False), key="show_legend"
        )
        main_plot_title = st.text_input(
            "Main Plot Title", value=st.session_state.get("main_plot_title", "My Plot Title"), key="main_plot_title"
        )
        x_label = st.text_input(
            "X-Axis Label", value=st.session_state.get("x_label", "Temperature (K)"), key="x_label"
        )
        y_label_left = st.text_input(
            "Left Y-Axis Label", value=st.session_state.get("y_label_left", "Resistance (Ω)"), key="y_label_left"
        )
        y_label_right = st.text_input(
            "Right Y-Axis Label", value=st.session_state.get("y_label_right", "Resistance (Ω)"), key="y_label_right"
        )
        calc_mob_den = st.checkbox("Automatically calculate Mobility & Density", key="calc_mob_den_checkbox")
        computed_legend = ""
        if calc_mob_den:
            geo_factor = st.number_input(
                "Geometric Factor", value=st.session_state.get("geo_factor_input", 1.0), format="%.4f", key="geo_factor_input"
            )
            field_low = st.number_input(
                "Field Cutoff Low",
                value=st.session_state.get("field_low_input", float(np.min(x_values))),
                format="%.2f",
                key="field_low_input",
            )
            field_high = st.number_input(
                "Field Cutoff High",
                value=st.session_state.get("field_high_input", float(np.max(x_values))),
                format="%.2f",
                key="field_high_input",
            )

            def get_channel_data(channel_key):
                idx = y_axis_options.index(channel_key)
                ch_info = st.session_state["selected_channels"][idx]
                data = data_utils.load_dataset(ch_info["file_path"], ch_info["data_path"])
                if data is None:
                    return None
                if data.ndim == 3 and data.shape[2] == 1:
                    data = data.squeeze(axis=2)
                if data.ndim == 1:
                    return data
                if ch_info["channel_index"] < data.shape[1]:
                    return data[:, ch_info["channel_index"]]
                return None

            if left_axis_choices:
                rxy_channel = left_axis_choices[0]
                rxy_data = get_channel_data(rxy_channel)
            else:
                rxy_data = None
            if left_axis_choices:
                rxx_channel = left_axis_choices[0]
                ryy_channel = left_axis_choices[0] if len(left_axis_choices) == 1 else left_axis_choices[1]
            else:
                rxx_channel = ryy_channel = None
            if rxy_data is not None and rxx_channel is not None:
                try:
                    density_val, density_err, _ = data_utils.extract_density(x_values, rxy_data, (field_low, field_high))
                    rxx_data = get_channel_data(rxx_channel)
                    ryy_data = get_channel_data(ryy_channel)
                    mob_vals = data_utils.extract_mobility(x_values, rxx_data, ryy_data, density_val, geo_factor)
                    mu_xx_cm = mob_vals[0] * 1e4
                    mu_yy_cm = mob_vals[1] * 1e4
                    density_cm = density_val * 1e-4
                    density_err_cm = density_err * 1e-4
                    mob_text = f"Mobility: μxx = {mu_xx_cm:.3g} cm²/Vs, μyy = {mu_yy_cm:.3g} cm²/Vs"
                    dens_text = f"Carrier Density: n = {density_cm:.3g} cm⁻² (± {density_err_cm:.3g})"
                    computed_legend = mob_text + "\n" + dens_text
                except Exception as e:
                    computed_legend = f"Error in Mobility/Density calculation: {e}"
            else:
                computed_legend = "Insufficient channel data for Mobility/Density calculation."
            if st.button("Update Legend Text", key="update_legend_text"):
                st.session_state["extra_legend_text"] = computed_legend
        if "extra_legend_text" not in st.session_state:
            st.session_state["extra_legend_text"] = ""
        extra_legend_text = st.text_area(
            "Additional Legend Info",
            value=st.session_state.get("extra_legend_text", ""),
            key="extra_legend_text_area",
        )
    st.write("### Step D: Plot Settings")
    x_nonnan = x_values[~np.isnan(x_values)]
    if x_nonnan.size == 0:
        st.error("All X-values are NaN; cannot specify data range.")
        return
    with st.expander("Data range", expanded=False):
        x_min_default = float(np.min(x_nonnan))
        x_max_default = float(np.max(x_nonnan))
        user_x_min = st.number_input(
            "X-Min",
            value=st.session_state.get("plot_xmin", x_min_default),
            format="%.2f",
            key="plot_xmin",
        )
        user_x_max = st.number_input(
            "X-Max",
            value=st.session_state.get("plot_xmax", x_max_default),
            format="%.2f",
            key="plot_xmax",
        )
        if user_x_min >= user_x_max:
            st.error("X-Min must be less than X-Max.")
            return
    with st.expander("Figure size and Font settings", expanded=False):
        fig_width = st.number_input(
            "Figure Width", 5.0, 30.0, st.session_state.get("fig_width", 12.0), 0.5, key="fig_width"
        )
        fig_height = st.number_input(
            "Figure Height", 4.0, 30.0, st.session_state.get("fig_height", 6.0), 0.5, key="fig_height"
        )
        title_font_size = st.number_input(
            "Title Font Size", 8, 32, st.session_state.get("title_font_size", 16), 1, key="title_font_size"
        )
        axis_label_font_size = st.number_input(
            "Axis Label Font Size", 8, 24, st.session_state.get("axis_label_font_size", 14), 1, key="axis_label_font_size"
        )
        tick_label_font_size = st.number_input(
            "Tick Label Font Size", 8, 20, st.session_state.get("tick_label_font_size", 12), 1, key="tick_label_font_size"
        )
        bold_title = st.checkbox(
            "Bold Title", value=st.session_state.get("bold_title", False), key="bold_title"
        )
        bold_axis_labels = st.checkbox(
            "Bold Axis Labels", value=st.session_state.get("bold_axis_labels", False), key="bold_axis_labels"
        )
    st.write("### Step E: Cut Data (Per Channel)")

    def get_channel_valid_x_count(ch_key):
        idx_c = y_axis_options.index(ch_key)
        ch_info = st.session_state["selected_channels"][idx_c]
        data_ = data_utils.load_dataset(ch_info["file_path"], ch_info["data_path"])
        if data_ is None:
            return 0
        if data_.ndim == 3 and data_.shape[2] == 1:
            data_ = data_.squeeze(axis=2)
        if data_.ndim == 1:
            y_vals = data_
        else:
            if ch_info["channel_index"] >= data_.shape[1]:
                return 0
            y_vals = data_[:, ch_info["channel_index"]]
        final_mask = (~np.isnan(x_values)) & (~np.isnan(y_vals)) & (
            (x_values >= user_x_min) & (x_values <= user_x_max)
        )
        return np.count_nonzero(final_mask)

    cut_map = {}
    all_channels = left_axis_choices + right_axis_choices
    for c in all_channels:
        idx_c = y_axis_options.index(c)
        ch_info = st.session_state["selected_channels"][idx_c]
        label_for_expander = ch_info.get(
            "renamed_name", f"{ch_info['file_name']} - {ch_info['channel_name']}"
        )
        valid_count = get_channel_valid_x_count(c)
        with st.expander(f"Cut Data for: {label_for_expander}", expanded=False):
            st.write(f"Number of valid X-values (non-NaN, after range filtering): {valid_count}")
            c_start = st.number_input(
                "Start index to cut out",
                min_value=0,
                value=st.session_state.get(f"cut_start_idx_{c}", 0),
                step=1,
                key=f"cut_start_idx_{c}",
            )
            c_end = st.number_input(
                "End index to cut out (inclusive)",
                min_value=0,
                value=st.session_state.get(f"cut_end_idx_{c}", 0),
                step=1,
                key=f"cut_end_idx_{c}",
            )
            cut_map[c] = (c_start, c_end)
    st.write("### Step F: Transformations & Scale per Channel")

    def reflect_x_local(xx, yy):
        if xx.size > 0:
            x_mid = 0.5 * (np.min(xx) + np.max(xx))
            x_ref = 2.0 * x_mid - xx
            sort_idx = np.argsort(x_ref)
            return x_ref[sort_idx], yy[sort_idx]
        return xx, yy

    def reflect_y_local(yy):
        if yy.size > 0:
            y_mid = 0.5 * (np.min(yy) + np.max(yy))
            return 2.0 * y_mid - yy
        return yy

    def apply_transformations(x_arr, y_arr, flip_x, flip_y, reflect_x, reflect_y, scale):
        x_local = x_arr.copy()
        y_local = y_arr.copy()
        if flip_x:
            y_local = -y_local
        if flip_y:
            x_local = -x_local
        if reflect_x:
            x_local, y_local = reflect_x_local(x_local, y_local)
        if reflect_y:
            y_local = reflect_y_local(y_local)
        y_local *= scale
        return x_local, y_local

    transforms_map = {}
    for c in all_channels:
        idx_c = y_axis_options.index(c)
        ch_info = st.session_state["selected_channels"][idx_c]
        label_for_expander = ch_info.get(
            "renamed_name", f"{ch_info['file_name']} - {ch_info['channel_name']}"
        )
        with st.expander(f"Transform: {label_for_expander}", expanded=False):
            flip_x_c = st.checkbox(
                "Flip X-axis (Invert Y)", value=st.session_state.get(f"flipx_{c}", False), key=f"flipx_{c}"
            )
            flip_y_c = st.checkbox(
                "Flip Y-axis (Invert X)", value=st.session_state.get(f"flipy_{c}", False), key=f"flipy_{c}"
            )
            reflect_x_c = st.checkbox(
                "Reflect X about midpoint", value=st.session_state.get(f"reflx_{c}", False), key=f"reflx_{c}"
            )
            reflect_y_c = st.checkbox(
                "Reflect Y about midpoint", value=st.session_state.get(f"refly_{c}", False), key=f"refly_{c}"
            )
            scale_factor_c = st.number_input(
                "Scale Factor (Y)",
                min_value=0.0,
                value=st.session_state.get(f"scale_{c}", 1.0),
                step=0.1,
                format="%.2f",
                key=f"scale_{c}",
            )
            transforms_map[c] = {
                "flip_x": flip_x_c,
                "flip_y": flip_y_c,
                "reflect_x": reflect_x_c,
                "reflect_y": reflect_y_c,
                "scale": scale_factor_c,
            }
    st.write("### Step G: Y-Axis Channel Style")
    default_left_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    styles_map = {}
    left_color_index = 0
    for c in all_channels:
        idx_c = y_axis_options.index(c)
        ch_info = st.session_state["selected_channels"][idx_c]
        label_for_style = ch_info.get(
            "renamed_name", f"{ch_info['file_name']} - {ch_info['channel_name']}"
        )
        default_color = (
            default_left_colors[left_color_index % len(default_left_colors)] if c in left_axis_choices else "#FF0000"
        )
        if c in left_axis_choices:
            left_color_index += 1
        with st.expander(f"Style: {label_for_style}", expanded=False):
            line_style = st.selectbox("Line Style", options=["-", "--", "-.", ":"], key=f"linestyle_{c}")
            marker_style = st.selectbox("Marker", options=["None", "o", "x", "s", "d", "^"], key=f"marker_{c}")
            picked_color = st.color_picker(
                "Color", value=st.session_state.get(f"color_{c}", default_color), key=f"color_{c}"
            )
            styles_map[c] = {
                "linestyle": line_style,
                "marker": "" if marker_style == "None" else marker_style,
                "color": picked_color,
            }
    if st.button("Generate Plot", key="generate_plot"):
        mask_nan = ~np.isnan(x_values)
        mask_range = (x_values >= user_x_min) & (x_values <= user_x_max)
        combined_mask = mask_nan & mask_range
        if not np.any(combined_mask):
            st.error("No data points found in the specified X range.")
            return
        fig, ax_left = plt.subplots(figsize=(fig_width, fig_height))

        def cut_data_by_indices(xx, yy, channel_key):
            n_points = xx.size
            if n_points == 0:
                return xx, yy
            c_start, c_end = cut_map.get(channel_key, (0, 0))
            c_start = min(max(c_start, 0), n_points - 1)
            c_end = min(max(c_end, 0), n_points - 1)
            if c_start <= c_end:
                xx = np.concatenate([xx[:c_start], xx[c_end + 1 :]])
                yy = np.concatenate([yy[:c_start], yy[c_end + 1 :]])
            return xx, yy

        for c in left_axis_choices:
            idx_c = y_axis_options.index(c)
            ch_info = st.session_state["selected_channels"][idx_c]
            loaded_data = data_utils.load_dataset(ch_info["file_path"], ch_info["data_path"])
            if loaded_data is None:
                continue
            if loaded_data.ndim == 3 and loaded_data.shape[2] == 1:
                loaded_data = loaded_data.squeeze(axis=2)
            y_vals = loaded_data if loaded_data.ndim == 1 else loaded_data[:, ch_info["channel_index"]]
            tr = transforms_map[c]
            x_loc, y_loc = apply_transformations(
                x_values,
                y_vals,
                flip_x=tr["flip_x"],
                flip_y=tr["flip_y"],
                reflect_x=tr["reflect_x"],
                reflect_y=tr["reflect_y"],
                scale=tr["scale"],
            )
            final_mask = combined_mask & ~np.isnan(y_loc)
            xx, yy = x_loc[final_mask], y_loc[final_mask]
            xx, yy = cut_data_by_indices(xx, yy, c)
            if xx.size == 0:
                st.error("No data remain after index cutting.")
                return
            style_info = styles_map.get(c, {})
            ax_left.plot(
                xx,
                yy,
                label=ch_info.get("renamed_name", c),
                linestyle=style_info.get("linestyle", "-"),
                marker=style_info.get("marker", ""),
                color=style_info.get("color", "#1f77b4"),
            )
        ax_left.set_xlabel(
            x_label, fontsize=axis_label_font_size, fontweight="bold" if bold_axis_labels else "normal"
        )
        ax_left.set_ylabel(
            y_label_left, fontsize=axis_label_font_size, fontweight="bold" if bold_axis_labels else "normal"
        )
        ax_left.tick_params(axis="both", labelsize=tick_label_font_size)
        ax_left.grid(True)
        if right_axis_choices:
            ax_right = ax_left.twinx()
            first_right = right_axis_choices[0]
            right_axis_color = styles_map.get(first_right, {}).get("color", "red")
            ax_right.spines["right"].set_color(right_axis_color)
            ax_right.tick_params(axis="y", labelsize=tick_label_font_size, colors=right_axis_color)
            for c in right_axis_choices:
                idx_c = y_axis_options.index(c)
                ch_info = st.session_state["selected_channels"][idx_c]
                loaded_data = data_utils.load_dataset(ch_info["file_path"], ch_info["data_path"])
                if loaded_data is None:
                    continue
                if loaded_data.ndim == 3 and loaded_data.shape[2] == 1:
                    loaded_data = loaded_data.squeeze(axis=2)
                y_vals = loaded_data if loaded_data.ndim == 1 else loaded_data[:, ch_info["channel_index"]]
                tr = transforms_map[c]
                x_loc, y_loc = apply_transformations(
                    x_values,
                    y_vals,
                    flip_x=tr["flip_x"],
                    flip_y=tr["flip_y"],
                    reflect_x=tr["reflect_x"],
                    reflect_y=tr["reflect_y"],
                    scale=tr["scale"],
                )
                final_mask = combined_mask & ~np.isnan(y_loc)
                xx, yy = x_loc[final_mask], y_loc[final_mask]
                xx, yy = cut_data_by_indices(xx, yy, c)
                if xx.size == 0:
                    st.error("No data remain after index cutting.")
                    return
                style_info = styles_map.get(c, {})
                ax_right.plot(
                    xx,
                    yy,
                    label=ch_info.get("renamed_name", c),
                    linestyle=style_info.get("linestyle", "-"),
                    marker=style_info.get("marker", ""),
                    color=style_info.get("color", "red"),
                )
            ax_right.set_ylabel(
                y_label_right,
                fontsize=axis_label_font_size,
                fontweight="bold" if bold_axis_labels else "normal",
                color=right_axis_color,
            )
        ax_left.set_title(
            main_plot_title, fontsize=title_font_size, fontweight="bold" if bold_title else "normal"
        )
        if show_legend:
            handles_left, labels_left = ax_left.get_legend_handles_labels()
            handles_right, labels_right = (
                ax_right.get_legend_handles_labels() if right_axis_choices else ([], [])
            )
            combined_handles = handles_left + handles_right
            combined_labels = labels_left + labels_right
            if extra_legend_text.strip():
                from matplotlib.lines import Line2D

                lines_extra = extra_legend_text.strip().split("\n")
                placeholders = [Line2D([0], [0], color="none") for _ in lines_extra]
                combined_handles += placeholders
                combined_labels += lines_extra
            ax_left.legend(combined_handles, combined_labels, loc="best", fontsize=12, frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        ax = fig.axes[0] if fig.axes else None
        title = ax.get_title() if ax else ""
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", title).strip("_") or "plot"
        filename = f"{safe}.png"
        buf = io.BytesIO()
        fig.savefig(buf, format="png", transparent=True)
        buf.seek(0)
        st.download_button(label="Download Plot as PNG", data=buf, file_name=filename, mime="image/png")

        def make_unique_plot_name(base_name):
            existing_names = [p["plot_name"] for p in st.session_state.get("plots_history", [])]
            if base_name not in existing_names:
                return base_name
            i = 2
            new_name = f"{base_name} ({i})"
            while new_name in existing_names:
                i += 1
                new_name = f"{base_name} ({i})"
            return new_name

        final_plot_name = make_unique_plot_name(main_plot_title)
        if "plots_history" not in st.session_state:
            st.session_state["plots_history"] = []
        st.session_state["plots_history"].insert(0, {"plot_name": final_plot_name, "figure": fig})
        st.success(f"Plot '{final_plot_name}' generated and stored in sidebar history!")


def main():
    session.initialize_session_state()
    st.set_page_config(page_title="🗂️ HDF5 Structure Dashboard", layout="wide")

    display_plot_history_sidebar()

    st.title("🗂️ HDF5 Files Structure Dashboard")
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.sidebar.header("Upload HDF5 Files")
    uploaded_files = st.sidebar.file_uploader(
        "Choose HDF5 files", type=["h5", "hdf5"], accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 {uploaded_file.name}", expanded=False):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                        st.session_state["temp_files"][uploaded_file.name] = tmp_path
                        logging.debug(f"Uploaded file saved temporarily at: {tmp_path}")
                    with h5py.File(tmp_path, "r") as f:
                        comment = f.attrs.get("comment", "No comment available.")
                        st.markdown("**File Notes:**")
                        st.markdown(
                            f"""<div style='height:200px; overflow:auto; border:1px solid #ccc; padding:10px;'>{comment}</div>""",
                            unsafe_allow_html=True,
                        )
                        structure = data_utils.parse_hdf5_structure(f)
                        if "Data" in structure:
                            data_structure = structure["Data"]
                            display_structure(
                                data_structure, uploaded_file.name, tmp_path, parent_path="Data", indent_level=0
                            )
                        else:
                            st.warning("The uploaded HDF5 file does not contain a 'Data' group.")
                except OSError:
                    st.error(
                        f"Could not open {uploaded_file.name}. It might be corrupted or not a valid HDF5 file."
                    )
                except Exception as e:
                    st.error(f"An unexpected error occurred while reading {uploaded_file.name}: {e}")

    if st.session_state.get("selected_dataset"):
        file_name, file_path, dataset_path = st.session_state["selected_dataset"]
        st.sidebar.header("View Selected Dataset")
        st.sidebar.write(f"**File:** {file_name}")
        st.sidebar.write(f"**Dataset Path:** {dataset_path}")
        try:
            display_dataset_content(file_name, file_path, dataset_path)
        except Exception as e:
            st.error(f"Error processing selected dataset: {e}")

    display_selected_channels()
    display_combined_data()
    plot_selected_channels()

    if st.sidebar.button("Clear All Selected Channels", key="clear_channels_button"):
        session.clear_selected_channels()

    with st.container():
        st.markdown("---")
        st.header("Session Management")
        st.text_input("Custom Session Name (optional)", key="custom_session_name")

        if st.button("Save Session", key="save_session_bottom"):
            session.save_session()
        if not os.path.exists(session.SESSIONS_FOLDER):
            os.makedirs(session.SESSIONS_FOLDER)
        session_list = os.listdir(session.SESSIONS_FOLDER)
        selected_session = st.selectbox(
            "Load Session", [""] + session_list, key="load_session_select_bottom"
        )
        if st.button("Load Selected Session", key="load_session_button_bottom"):
            if selected_session:
                session.load_session(selected_session)
            else:
                st.warning("Select a session to load.")
        if st.button("Update Current Session", key="update_session_button_bottom"):
            if st.session_state.get("current_session_name"):
                session.update_session(st.session_state["current_session_name"])
            else:
                st.warning("No current session to update. Save a session first.")
