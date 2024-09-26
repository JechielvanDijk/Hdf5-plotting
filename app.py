# app.py

import streamlit as st
import h5py
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
import logging




# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_hdf5_structure(file):
    """
    Recursively parses the structure of an HDF5 file.

    Args:
        file (h5py.File): An open HDF5 file object.

    Returns:
        dict: A nested dictionary representing the file structure.
    """
    structure = {}
    logging.debug("Starting to parse HDF5 file structure.")

    def visit(name, obj):
        logging.debug(f"Visiting: {name}")
        path = name.split('/')
        current = structure
        for part in path:
            if part:
                current = current.setdefault(part, {})
        if isinstance(obj, h5py.Dataset):
            current['_type'] = 'Dataset'
            current['_shape'] = obj.shape
            current['_dtype'] = str(obj.dtype)
            logging.debug(f"Added Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            current['_type'] = 'Group'
            logging.debug(f"Added Group: {name}")

    file.visititems(visit)
    logging.debug("Completed parsing HDF5 file structure.")
    return structure

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'selected_dataset' not in st.session_state:
        st.session_state['selected_dataset'] = None
    if 'selected_channels' not in st.session_state:
        st.session_state['selected_channels'] = []
    if 'temp_files' not in st.session_state:
        st.session_state['temp_files'] = {}


def display_structure(structure, file_name, file_path, parent_path="", indent_level=0):
    """
    Recursively displays the HDF5 structure using Streamlit expanders and buttons.

    Args:
        structure (dict): The HDF5 file structure.
        file_name (str): Name of the HDF5 file.
        file_path (str): Path to the temporary HDF5 file.
        parent_path (str): Current path in the HDF5 hierarchy.
        indent_level (int): Current indentation level.
    """
    logging.debug(f"Displaying structure for file: {file_name}, path: {parent_path}")

    for key, value in structure.items():
        if key.startswith('_'):
            logging.debug(f"Skipping metadata key: {key}")
            continue  # Skip metadata keys

        current_path = f"{parent_path}/{key}" if parent_path else key
        logging.debug(f"Processing key: {key}, current_path: {current_path}")

        if isinstance(value, dict):
            if value.get('_type') == 'Dataset':
                # Display a button for the dataset with indentation
                button_key = f"{file_name}_{current_path}"
                logging.debug(f"Creating button for Dataset: {key} with key: {button_key}")

                if indent_level == 0:
                    # No indentation needed at top level
                    if st.button(f"{key}", key=button_key):
                        st.session_state['selected_dataset'] = (file_name, file_path, current_path)
                        logging.debug(f"Dataset selected: {file_name} - {current_path}")
                else:
                    # Calculate indentation width
                    indentation_width = indent_level * 0.05  # 5% per indent level
                    if indentation_width > 0.5:
                        indentation_width = 0.5  # Maximum indentation of 50%

                    remaining_width = 1.0 - indentation_width
                    if remaining_width <= 0:
                        remaining_width = 0.1  # Ensure there's space for the button

                    columns = st.columns([indentation_width, remaining_width])

                    with columns[1]:
                        if st.button(f"{key}", key=button_key):
                            st.session_state['selected_dataset'] = (file_name, file_path, current_path)
                            logging.debug(f"Dataset selected: {file_name} - {current_path}")
            else:
                # Display an expander for the group with indentation
                if indent_level == 0:
                    with st.expander(f"{key}", expanded=False):
                        display_structure(value, file_name, file_path, current_path, indent_level + 1)
                else:
                    # Calculate indentation width
                    indentation_width = indent_level * 0.05  # 5% per indent level
                    if indentation_width > 0.5:
                        indentation_width = 0.5  # Maximum indentation of 50%

                    remaining_width = 1.0 - indentation_width
                    if remaining_width <= 0:
                        remaining_width = 0.1  # Ensure there's space for the expander

                    columns = st.columns([indentation_width, remaining_width])

                    with columns[1]:
                        with st.expander(f"{key}", expanded=False):
                            display_structure(value, file_name, file_path, current_path, indent_level + 1)



def load_dataset(file_path, dataset_path):
    """
    Loads the dataset from the HDF5 file.

    Args:
        file_path (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset within the HDF5 file.

    Returns:
        numpy.ndarray: The dataset content.
    """
    logging.debug(f"Loading dataset from file: {file_path}, dataset_path: {dataset_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            dataset = f[dataset_path][()]
        logging.debug(f"Loaded dataset shape: {dataset.shape}, dtype: {dataset.dtype}")
        return dataset
    except KeyError:
        st.error(f"Dataset path '{dataset_path}' not found in the HDF5 file.")
        logging.error(f"Dataset path '{dataset_path}' not found in the HDF5 file.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        logging.exception("Exception occurred while loading dataset.")
        return None


def add_selected_channels(selected_channels):
    """
    Adds selected channels to the session state.

    Args:
        selected_channels (list): List of selected channel dictionaries.
    """
    st.session_state['selected_channels'].extend(selected_channels)
    logging.debug(f"Added channels to session state: {selected_channels}")


def remove_selected_channel(index):
    """
    Removes a selected channel from the session state.

    Args:
        index (int): Index of the channel to remove.
    """
    try:
        channel = st.session_state['selected_channels'].pop(index)
        logging.debug(f"Removed channel from session state: {channel}")
        # Check if any other channel uses the same file
        file_path = channel['file_path']
        if not any(ch['file_path'] == file_path for ch in st.session_state['selected_channels']):
            # Delete the temporary file
            try:
                os.unlink(file_path)
                file_name = channel['file_name']
                if file_name in st.session_state['temp_files']:
                    del st.session_state['temp_files'][file_name]
                logging.debug(f"Deleted temporary file: {file_path}")
            except FileNotFoundError:
                logging.warning(f"Temporary file already deleted: {file_path}")
            except Exception as e:
                st.error(f"Error deleting temporary file {file_path}: {e}")
                logging.error(f"Error deleting temporary file {file_path}: {e}")
    except IndexError:
        st.error("Invalid channel index.")
        logging.error("Attempted to remove a channel with an invalid index.")


def clear_selected_channels():
    """
    Clears all selected channels and deletes associated temporary files.
    """
    for channel in st.session_state['selected_channels']:
        file_path = channel['file_path']
        try:
            os.unlink(file_path)
            logging.debug(f"Deleted temporary file: {file_path}")
        except FileNotFoundError:
            logging.warning(f"Temporary file already deleted: {file_path}")
        except Exception as e:
            st.error(f"Error deleting temporary file {file_path}: {e}")
            logging.error(f"Error deleting temporary file {file_path}: {e}")
    st.session_state['selected_channels'] = []
    st.session_state['temp_files'] = {}
    st.success("All selected channels have been cleared and temporary files deleted.")
    logging.debug("Cleared all selected channels and deleted temporary files.")


def display_selected_channels():
    """
    Displays the list of selected channels with options to remove individual channels.
    """
    if not st.session_state['selected_channels']:
        st.info("No channels have been selected yet.")
        return

    st.sidebar.header("Selected Channels")
    for idx, channel in enumerate(st.session_state['selected_channels'], start=1):
        col1, col2 = st.sidebar.columns([4, 1])
        col1.write(f"{idx}. **{channel['file_name']} - {channel['channel_name']}**")
        if col2.button("Remove", key=f"remove_{idx}"):
            remove_selected_channel(idx - 1)
            st.experimental_rerun()


def display_combined_data():
    """
    Displays the combined data of all selected channels from different files,
    with numbers formatted to display significant digits appropriately.
    """
    if not st.session_state['selected_channels']:
        st.info("No channels have been selected yet.")
        return

    st.write("## Combined Data of Selected Channels")
    data_frames = []
    for channel in st.session_state['selected_channels']:
        try:
            # Load data
            data = load_dataset(channel['file_path'], channel['data_path'])
            logging.debug(f"Loaded data for channel: {channel}")

            if data is None:
                continue  # Skip if data failed to load

            # Handle 3D data
            if data.ndim == 3 and data.shape[2] == 1:
                data = data.squeeze(axis=2)

            # Check if data is 1D
            if data.ndim == 1:
                df_data = pd.DataFrame(data, columns=[f"{channel['file_name']} - {channel['channel_name']}"])
                data_frames.append(df_data)
                logging.debug(f"Added 1D data for channel: {channel['channel_name']}")
                continue  # Move to the next channel

            # Convert to DataFrame
            df_data = pd.DataFrame(data)
            logging.debug(f"DataFrame shape before selecting column: {df_data.shape}")

            # Adjust for zero-based indexing
            column_index = channel['channel_index']
            if column_index >= df_data.shape[1]:
                st.error(f"Channel index {column_index} out of bounds for file `{channel['file_name']}`.")
                logging.error(f"Channel index {column_index} out of bounds for file `{channel['file_name']}`.")
                continue

            # Select the specific column using iloc to avoid KeyError
            df_data = df_data.iloc[:, [column_index]]
            df_data.columns = [f"{channel['file_name']} - {channel['channel_name']}"]
            data_frames.append(df_data)
            logging.debug(f"Added 2D data for channel: {channel['channel_name']}")

        except Exception as e:
            st.error(f"Error loading data for channel `{channel['channel_name']}` from `{channel['file_name']}`: {e}")
            logging.exception("Exception occurred while loading channel data.")

    if data_frames:
        # Determine the maximum length among all data frames
        max_length = max(df.shape[0] for df in data_frames)
        # Pad shorter data frames with NaN to align data
        padded_data_frames = [df.reindex(range(max_length)) for df in data_frames]
        # Concatenate data frames horizontally
        combined_data = pd.concat(padded_data_frames, axis=1)

        # Format numbers in the DataFrame
        def format_numbers(val):
            if pd.isnull(val):
                return ''
            elif val == 0:
                return '0.00'
            else:
                return f"{val:.4g}"

        formatted_combined_data = combined_data.applymap(format_numbers)
        st.dataframe(formatted_combined_data)
        logging.debug("Displayed combined data for selected channels with formatted numbers.")
    else:
        st.error("No data available to display.")



def plot_selected_channels():
    """
    Plots selected channels on the same axes within a single figure.
    Users can scale, reverse, and customize the plot style for each Y-axis data column.
    """
    if not st.session_state['selected_channels']:
        st.info("No channels have been selected to plot.")
        return

    st.write("## Plot of Selected Channels")

    # Select X-axis
    st.write("### Select X-axis Channel")
    x_axis_options = [
        f"{idx + 1}. {ch['file_name']} - {ch['channel_name']}"
        for idx, ch in enumerate(st.session_state['selected_channels'])
    ]
    x_axis_choice = st.selectbox("Choose X-axis Channel", options=x_axis_options)

    # Find the selected X-axis channel
    try:
        x_idx = x_axis_options.index(x_axis_choice)
    except ValueError:
        st.error("Selected X-axis channel is invalid.")
        logging.error("Selected X-axis channel is invalid.")
        return

    x_channel = st.session_state['selected_channels'][x_idx]
    try:
        x_data = load_dataset(x_channel['file_path'], x_channel['data_path'])
        if x_data is None:
            return  # Skip if data failed to load
        if x_channel['channel_index'] >= x_data.shape[1]:
            st.error(
                f"Channel index {x_channel['channel_index']} out of bounds for file `{x_channel['file_name']}`."
            )
            logging.error(
                f"Channel index {x_channel['channel_index']} out of bounds for file `{x_channel['file_name']}`."
            )
            return
        if x_data.ndim == 3 and x_data.shape[2] == 1:
            x_data = x_data.squeeze(axis=2)
        x_values = x_data[:, x_channel['channel_index']]
        logging.debug(f"X-axis data shape: {x_values.shape}")
    except Exception as e:
        st.error(f"Error loading X-axis data: {e}")
        logging.exception("Exception occurred while loading X-axis data.")
        return

    # Select Y-axis channels
    st.write("### Select Y-axis Channels")
    y_axis_options = x_axis_options.copy()
    y_axis_choices = st.multiselect(
        "Choose Y-axis Channels", options=y_axis_options, default=[]
    )

    if not y_axis_choices:
        st.info("Select at least one Y-axis channel to plot.")
        return

    # Assign Y-axis channels to right y-axis (optional)
    st.write("### Assign Y-axis Channels to Right Axis (Optional)")
    right_axis_choices = st.multiselect(
        "Channels for Right Y-Axis", options=y_axis_choices, default=[]
    )
    left_axis_choices = [choice for choice in y_axis_choices if choice not in right_axis_choices]

    if not left_axis_choices and not right_axis_choices:
        st.error(
            "At least one Y-axis channel must be assigned to either the left or right axis."
        )
        return

    # Collect Y-axis data and allow renaming for each channel
    y_channels_left = []
    y_data_list_left = []
    custom_legend_names_left = []
    y_channels_right = []
    y_data_list_right = []
    custom_legend_names_right = []

    # Legend, plot title, and axis labels
    st.write("### Legend and Custom Plot Names")
    with st.expander("Customize Legend and Plot Names", expanded=False):
        show_legend = st.checkbox("Show Legend", value=True)
        plot_title = st.text_input("Plot Title", value="Plot of Selected Channels")
        x_axis_label = st.text_input("X-Axis Label", value="X Values")
        y_axis_label_left = st.text_input("Left Y-Axis Label", value="Left Y-Axis")
        y_axis_label_right = st.text_input("Right Y-Axis Label", value="Right Y-Axis (Optional)")

        # Custom names for left Y-axis channels
        for choice in left_axis_choices:
            custom_name = st.text_input(
                f"Custom name for Left Y-axis channel '{choice}'", value=choice
            )
            custom_legend_names_left.append(custom_name)

        # Custom names for right Y-axis channels
        for choice in right_axis_choices:
            custom_name = st.text_input(
                f"Custom name for Right Y-axis channel '{choice}'", value=choice
            )
            custom_legend_names_right.append(custom_name)

    # Second Legend (Additional text box)
    with st.expander("Second Legend", expanded=False):
        additional_legend_text = st.text_area("Enter additional legend text:")

    # Data Transformation Options
    st.write("### Data Transformation Options")
    with st.expander("Transform Y-axis Data", expanded=False):
        scaling_factors = {}
        reverse_flags = {}

        st.write("#### Left Y-Axis Channels")
        for idx, choice in enumerate(left_axis_choices):
            custom_name = custom_legend_names_left[idx]  # Map to custom name
            scaling_factor = st.number_input(
                f"Scaling factor for '{custom_name}'",
                value=1.0,
                step=0.1,
                format="%.2f",
                key=f"scale_left_{idx}",
            )
            reverse = st.checkbox(
                f"Reverse data for '{custom_name}'", value=False, key=f"reverse_left_{idx}"
            )
            scaling_factors[choice] = scaling_factor
            reverse_flags[choice] = reverse

        st.write("#### Right Y-Axis Channels")
        for idx, choice in enumerate(right_axis_choices):
            custom_name = custom_legend_names_right[idx]  # Map to custom name
            scaling_factor = st.number_input(
                f"Scaling factor for '{custom_name}'",
                value=1.0,
                step=0.1,
                format="%.2f",
                key=f"scale_right_{idx}",
            )
            reverse = st.checkbox(
                f"Reverse data for '{custom_name}'", value=False, key=f"reverse_right_{idx}"
            )
            scaling_factors[choice] = scaling_factor
            reverse_flags[choice] = reverse

    # Helper function to apply scaling and reversing
    def transform_data(y, scale=1.0, reverse=False):
        y = y * scale
        if reverse:
            y = np.flip(y)
        return y

    # Helper function to load y-axis data
    def load_y_data(choice_list, y_channels, y_data_list, scaling_factors, reverse_flags):
        for choice in choice_list:
            try:
                idx = y_axis_options.index(choice)
            except ValueError:
                st.error(f"Selected Y-axis channel `{choice}` is invalid.")
                logging.error(f"Selected Y-axis channel `{choice}` is invalid.")
                continue

            channel = st.session_state['selected_channels'][idx]
            try:
                data = load_dataset(channel['file_path'], channel['data_path'])
                if data is None:
                    continue  # Skip if data failed to load

                # Handle 3D data
                if data.ndim == 3 and data.shape[2] == 1:
                    data = data.squeeze(axis=2)

                # Check if data is 1D
                if data.ndim == 1:
                    y_values = data
                else:
                    column_index = channel['channel_index']
                    if column_index >= data.shape[1]:
                        st.error(
                            f"Channel index {column_index} out of bounds for file `{channel['file_name']}`."
                        )
                        logging.error(
                            f"Channel index {column_index} out of bounds for file `{channel['file_name']}`."
                        )
                        continue
                    y_values = data[:, column_index]

                # Apply transformations
                scale = scaling_factors.get(choice, 1.0)
                reverse = reverse_flags.get(choice, False)
                y_values = transform_data(y_values, scale=scale, reverse=reverse)

                y_channels.append(choice)
                y_data_list.append(y_values)
                logging.debug(
                    f"Loaded Y-axis data for channel: {channel['channel_name']} with scale={scale} and reverse={reverse}"
                )

            except Exception as e:
                st.error(
                    f"Error loading Y-axis data for channel `{channel['channel_name']}` from `{channel['file_name']}`: {e}"
                )
                logging.exception("Exception occurred while loading Y-axis data.")

    # Load data for left and right Y-axes
    load_y_data(
        left_axis_choices,
        y_channels_left,
        y_data_list_left,
        scaling_factors,
        reverse_flags,
    )
    load_y_data(
        right_axis_choices,
        y_channels_right,
        y_data_list_right,
        scaling_factors,
        reverse_flags,
    )

    if not y_data_list_left and not y_data_list_right:
        st.error("No valid Y-axis channels to plot.")
        return

    # Plotting Section
    st.write("### Customize the Plot")

    # Automatically set min and max based on x_values
    with st.expander("Data Range", expanded=False):
        st.write("**Specify Data Range (X-Values)**")
        x_min_default = float(np.min(x_values))
        x_max_default = float(np.max(x_values))
        x_min = st.number_input("Minimum X-Value", value=x_min_default, format="%.2f")
        x_max = st.number_input("Maximum X-Value", value=x_max_default, format="%.2f")

    # Style settings
    st.write("### Style")

    # Figure Size
    with st.expander("Figure Size", expanded=False):
        fig_width = st.number_input(
            "Figure Width (in inches)",
            min_value=5.0,
            max_value=30.0,
            value=15.0,
            step=0.5,
        )
        fig_height = st.number_input(
            "Figure Height (in inches)",
            min_value=5.0,
            max_value=30.0,
            value=10.0,
            step=0.5,
        )

    # Font Settings
    with st.expander("Font Settings", expanded=False):
        title_font_size = st.number_input(
            "Title Font Size", min_value=8, max_value=32, value=16, step=1
        )
        axis_label_font_size = st.number_input(
            "Axis Label Font Size", min_value=8, max_value=24, value=14, step=1
        )
        tick_label_font_size = st.number_input(
            "Tick Label Font Size", min_value=8, max_value=20, value=12, step=1
        )
        bold_title = st.checkbox("Bold Title", value=False)
        bold_axis_labels = st.checkbox("Bold Axis Labels", value=False)

    # Style options for each plot
    st.write("**Style Options for Each Plot**")
    # Initialize dictionaries to store style settings
    plot_types = {}
    colors = {}
    line_styles = {}

    # For left Y-axis channels
    for idx, choice in enumerate(left_axis_choices):
        custom_name = custom_legend_names_left[idx]
        with st.expander(f"Style Options for '{custom_name}'", expanded=False):
            plot_type = st.selectbox(
                f"Plot Type for '{custom_name}'",
                options=['Line Plot', 'Scatter Plot'],
                index=0,
                key=f"plot_type_left_{idx}",
            )
            plot_types[choice] = plot_type

            color = st.color_picker(
                f"Color for '{custom_name}' (default is Matplotlib default color cycle)",
                key=f"color_left_{idx}",
                value=None,
            )
            colors[choice] = color

            line_style = st.selectbox(
                f"Line Style for '{custom_name}'",
                options=['Solid', 'Dashed', 'Dotted', 'Dash-dot'],
                index=0,
                key=f"line_style_left_{idx}",
            )
            line_styles[choice] = line_style

    # For right Y-axis channels
    for idx, choice in enumerate(right_axis_choices):
        custom_name = custom_legend_names_right[idx]
        with st.expander(f"Style Options for '{custom_name}'", expanded=False):
            plot_type = st.selectbox(
                f"Plot Type for '{custom_name}'",
                options=['Line Plot', 'Scatter Plot'],
                index=0,
                key=f"plot_type_right_{idx}",
            )
            plot_types[choice] = plot_type

            color = st.color_picker(
                f"Color for '{custom_name}' (default is red to match right Y-axis)",
                key=f"color_right_{idx}",
                value='#FF0000',  # Default to red
            )
            colors[choice] = color

            line_style = st.selectbox(
                f"Line Style for '{custom_name}'",
                options=['Solid', 'Dashed', 'Dotted', 'Dash-dot'],
                index=0,
                key=f"line_style_right_{idx}",
            )
            line_styles[choice] = line_style

    # Submit button
    if st.button("Generate Plot"):
        # Validate data range
        if x_min >= x_max:
            st.error("Minimum X-Value must be less than Maximum X-Value.")
        else:
            # Filter data based on x-values
            mask = (x_values >= x_min) & (x_values <= x_max)
            filtered_x = x_values[mask]
            if filtered_x.size == 0:
                st.error("No data points found in the specified X-Value range.")
                logging.debug(
                    "No data points found after filtering with the specified X-Value range."
                )
                st.stop()
            filtered_y_left = [y_data[mask] for y_data in y_data_list_left]
            filtered_y_right = [y_data[mask] for y_data in y_data_list_right]

            # Create plot
            fig, ax_left = plt.subplots(figsize=(fig_width, fig_height))

            # Get default color cycle
            default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            color_cycle_left = default_colors.copy()
            color_cycle_right = ['#FF0000']  # Default color for right Y-axis is red

            # Plot left Y-axis data
            for idx, (y, label) in enumerate(zip(filtered_y_left, custom_legend_names_left)):
                choice = left_axis_choices[idx]
                color = colors.get(choice)
                if not color or color == '#000000':  # If no color specified or black (default color_picker value)
                    # Use default color cycle
                    color = color_cycle_left[idx % len(color_cycle_left)]
                plot_type = plot_types.get(choice, 'Line Plot')
                line_style = line_styles.get(choice, 'Solid')
                linestyle_map = {'Solid': '-', 'Dashed': '--', 'Dotted': ':', 'Dash-dot': '-.'}
                linestyle = linestyle_map.get(line_style, '-')

                if plot_type == 'Scatter Plot':
                    ax_left.scatter(filtered_x, y, label=label, color=color)
                else:
                    ax_left.plot(filtered_x, y, label=label, color=color, linestyle=linestyle)

            # Customize left Y-axis
            ax_left.set_xlabel(
                x_axis_label,
                fontsize=axis_label_font_size,
                fontweight='bold' if bold_axis_labels else 'normal',
            )
            ax_left.set_ylabel(
                y_axis_label_left,
                fontsize=axis_label_font_size,
                fontweight='bold' if bold_axis_labels else 'normal',
            )
            ax_left.tick_params(axis='both', which='major', labelsize=tick_label_font_size)
            ax_left.grid(True)

            # Plot right Y-axis data if any
            if filtered_y_right:
                ax_right = ax_left.twinx()
                ax_right.spines['right'].set_color('red')  # Set the spine color to red
                for idx, (y, label) in enumerate(zip(filtered_y_right, custom_legend_names_right)):
                    choice = right_axis_choices[idx]
                    color = colors.get(choice)
                    if not color or color == '#000000':  # If no color specified or black (default color_picker value)
                        # Default to red to match right Y-axis
                        color = 'red'
                    plot_type = plot_types.get(choice, 'Line Plot')
                    line_style = line_styles.get(choice, 'Solid')
                    linestyle_map = {'Solid': '-', 'Dashed': '--', 'Dotted': ':', 'Dash-dot': '-.'}
                    linestyle = linestyle_map.get(line_style, '-')

                    if plot_type == 'Scatter Plot':
                        ax_right.scatter(filtered_x, y, label=label, color=color, marker='x')
                    else:
                        ax_right.plot(filtered_x, y, label=label, color=color, linestyle=linestyle)

                # Customize right Y-axis
                ax_right.set_ylabel(
                    y_axis_label_right,
                    fontsize=axis_label_font_size,
                    fontweight='bold' if bold_axis_labels else 'normal',
                    color='red',  # Set the axis label color to red
                )
                ax_right.tick_params(
                    axis='y', which='major', labelsize=tick_label_font_size, colors='red'
                )
                ax_right.spines['right'].set_color('red')  # Set the spine color to red

            # Set plot title
            ax_left.set_title(
                plot_title,
                fontsize=title_font_size,
                fontweight='bold' if bold_title else 'normal',
            )

            # Handling legends
            handles_left, labels_left = ax_left.get_legend_handles_labels()
            handles_right, labels_right = ([], [])
            if filtered_y_right:
                handles_right, labels_right = ax_right.get_legend_handles_labels()

            # Combine legends from both axes
            if show_legend:
                # Create main legend
                combined_handles = handles_left + handles_right
                combined_labels = labels_left + labels_right

                # If additional legend text is provided, create custom legend entries
                if additional_legend_text.strip():
                    # Split additional legend text into lines
                    legend_lines = additional_legend_text.strip().split('\n')
                    # Create custom legend entries with empty handles
                    from matplotlib.lines import Line2D
                    custom_lines = [Line2D([0], [0], color='none') for _ in legend_lines]
                    # Append to the combined legend
                    combined_handles += custom_lines
                    combined_labels += legend_lines

                # Place the combined legend
                ax_left.legend(
                    combined_handles,
                    combined_labels,
                    loc='best',
                    fontsize=12,
                    frameon=True
                )

            plt.tight_layout()
            st.pyplot(fig)
            logging.debug("Generated plot with left and right Y-axes.")

def display_dataset_content(file_name, file_path, dataset_path):
    """
    Displays the content of the selected dataset based on its type.
    If the selected dataset is "Data/Channel names", provides channel selection functionality.
    Numbers are formatted to display significant digits appropriately.

    Args:
        file_name (str): Name of the HDF5 file.
        file_path (str): Path to the HDF5 file.
        dataset_path (str): Path to the dataset within the HDF5 file.
    """
    try:
        data = load_dataset(file_path, dataset_path)
        if data is None:
            return  # Skip if data failed to load

        st.write(f"### 📄 Dataset: `{dataset_path}` in `{file_name}`")
        logging.debug(f"Displaying dataset content for: {dataset_path}")

        # Determine the type of data and display accordingly
        if isinstance(data, np.ndarray):
            if data.dtype.names:  # Check if the dtype is structured (has field names)
                logging.debug("Dataset has structured dtype.")
                # Convert structured array to pandas DataFrame
                df = pd.DataFrame(data)
                logging.debug(f"DataFrame created with shape: {df.shape}")

                # Decode byte strings in object columns
                for col in df.select_dtypes(include=['object']).columns:
                    logging.debug(f"Decoding byte strings in column: {col}")
                    df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

                # Check if the selected dataset is "Data/Channel names"
                if dataset_path == "Data/Channel names":
                    st.write("#### Select Channels by Selecting Rows in the Table Below")
                    logging.debug("Providing channel selection by row number.")

                    # Add a row number column
                    df.reset_index(inplace=True)
                    df.rename(columns={'index': 'Row Number'}, inplace=True)
                    logging.debug("Added 'Row Number' column to DataFrame.")

                    # Display the DataFrame with row numbers
                    st.dataframe(df)

                    # Provide a multiselect widget for row numbers
                    row_numbers = df['Row Number'].tolist()
                    selected_rows = st.multiselect(
                        f"Select Channel Rows from `{file_name}` (by Row Number)",
                        options=row_numbers,
                        default=[]
                    )
                    logging.debug(f"Selected row numbers: {selected_rows}")

                    if selected_rows:
                        # Extract selected channel names based on row numbers
                        selected_channels = df[df['Row Number'].isin(selected_rows)]
                        channel_info = [
                            {
                                'file_name': file_name,
                                'channel_name': row[df.columns[1]],  # Assuming channel names are in the second column
                                'file_path': file_path,
                                'data_path': "Data/Data",  # Adjust if necessary
                                'channel_index': row['Row Number']  # Zero-based indexing
                            }
                            for idx, row in selected_channels.iterrows()
                        ]
                        logging.debug(f"Selected channels: {channel_info}")

                        # Add selected channels to session state
                        if st.button("Add Selected Channels"):
                            add_selected_channels(channel_info)
                            st.success(f"Added {len(channel_info)} channels from `{file_name}`.")
                            logging.debug(f"Added channels to session state: {channel_info}")
                    else:
                        st.info("Select one or more channels from the list above to add to your selection.")
                        logging.debug("No channels selected.")
                else:
                    # For other structured datasets, format numbers and display the DataFrame
                    def format_numbers(val):
                        if pd.isnull(val):
                            return ''
                        elif isinstance(val, (int, float)):
                            if val == 0:
                                return '0.00'
                            else:
                                return f"{val:.4g}"
                        else:
                            return val  # Non-numeric values remain unchanged

                    formatted_df = df.applymap(format_numbers)
                    st.dataframe(formatted_df)
                    logging.debug("Displayed structured dataset as DataFrame with formatted numbers.")
            else:
                # Handle other data types as before
                logging.debug("Dataset does not have structured dtype.")
                if data.dtype.kind in {'i', 'f'}:
                    logging.info("Dataset is numerical.")
                    # Handle 3D numerical data by squeezing if necessary
                    if data.ndim == 3 and data.shape[2] == 1:
                        logging.debug("Squeezing last dimension of 3D numerical data.")
                        data = data.squeeze(axis=2)
                        logging.debug(f"New data shape after squeeze: {data.shape}")

                    if data.ndim <= 2:
                        max_rows = 100  # Limit for performance
                        if data.shape[0] > max_rows:
                            st.write(f"Displaying first {max_rows} rows:")
                            display_data = data[:max_rows, ...]
                        else:
                            display_data = data

                        # Convert to DataFrame for formatting
                        df = pd.DataFrame(display_data)
                        def format_numbers(val):
                            if pd.isnull(val):
                                return ''
                            elif val == 0:
                                return '0.00'
                            else:
                                return f"{val:.4g}"
                        formatted_df = df.applymap(format_numbers)
                        st.dataframe(formatted_df)
                        logging.debug(f"Displayed numerical data with shape: {display_data.shape}")

                        # Provide a line chart if 1D or a heatmap if 2D
                        if data.ndim == 1 or data.shape[1] == 1:
                            fig, ax = plt.subplots()
                            ax.plot(data.flatten())
                            ax.set_title(f"Line Plot of {dataset_path}")
                            st.pyplot(fig)
                            logging.debug(f"Plotted 1D data for {dataset_path}.")
                        elif data.ndim == 2:
                            fig, ax = plt.subplots()
                            cax = ax.imshow(data, aspect='auto', cmap='viridis')
                            fig.colorbar(cax)
                            ax.set_title(f"Heatmap of {dataset_path}")
                            st.pyplot(fig)
                            logging.debug(f"Plotted 2D data as heatmap for {dataset_path}.")
                    else:
                        st.write("Data format not supported for plotting.")
                        logging.warning("Unsupported data format for numerical data.")
                elif data.dtype.kind in {'S', 'U'}:
                    logging.info("Dataset contains strings.")
                    # Handle byte strings or Unicode strings
                    # Convert to regular strings if necessary
                    if data.ndim == 0:
                        # Scalar string
                        decoded_str = data.decode('utf-8') if isinstance(data, bytes) else data
                        st.text(decoded_str)
                        logging.debug("Displayed scalar string.")
                    else:
                        # Array of strings
                        decoded_str = [x.decode('utf-8') if isinstance(x, bytes) else x for x in data.flatten()]
                        st.write(decoded_str)
                        logging.debug("Displayed array of strings.")
                else:
                    st.write("Unsupported data type for preview.")
                    logging.warning("Unsupported data type encountered.")
    except Exception as e:
        st.error(f"An error occurred while displaying the dataset: {e}")
        logging.exception("Exception occurred in display_dataset_content.")


def main():
    initialize_session_state()

    st.set_page_config(page_title="🗂️ HDF5 Structure Dashboard", layout="wide")

    # Title
    st.title("🗂️ HDF5 Files Structure Dashboard")

    # Add vertical spacing between title and upload section
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.sidebar.header("Upload HDF5 Files")
    uploaded_files = st.sidebar.file_uploader(
        "Choose HDF5 files", type=["h5", "hdf5"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 {uploaded_file.name}", expanded=False):
                try:
                    # Save the uploaded file to a temporary location for h5py to read
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                        st.session_state['temp_files'][uploaded_file.name] = tmp_path
                        logging.debug(f"Uploaded file saved temporarily at: {tmp_path}")

                    with h5py.File(tmp_path, 'r') as f:
                        # Read the 'comment' attribute
                        comment = f.attrs.get('comment', 'No comment available.')

                        # Display the comment
                        st.markdown("**File Notes:**")
                        st.markdown(
                            f"<div style='height:200px; overflow:auto; border:1px solid #ccc; padding:10px;'>{comment}</div>",
                            unsafe_allow_html=True
                        )

                        # Proceed to parse and display the file structure
                        structure = parse_hdf5_structure(f)

                        # Display only the "Data" group
                        if 'Data' in structure:
                            data_structure = structure['Data']
                            display_structure(data_structure, uploaded_file.name, tmp_path, parent_path="Data", indent_level=0)
                            logging.debug(f"Structure displayed for 'Data' group in file: {uploaded_file.name}")
                        else:
                            st.warning("The uploaded HDF5 file does not contain a 'Data' group.")
                            logging.warning(f"'Data' group not found in file: {uploaded_file.name}")

                except OSError as e:
                    st.error(f"Could not open {uploaded_file.name}. It might be corrupted or not a valid HDF5 file.")
                    logging.error(f"Could not open {uploaded_file.name}: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred while reading {uploaded_file.name}: {e}")
                    logging.exception(f"Unexpected error with file {uploaded_file.name}.")

    # Handle dataset selection from session state
    if st.session_state['selected_dataset']:
        file_name, file_path, dataset_path = st.session_state['selected_dataset']
        st.sidebar.header("View Selected Dataset")
        st.sidebar.write(f"**File:** {file_name}")
        st.sidebar.write(f"**Dataset Path:** {dataset_path}")

        try:
            # Display the dataset content
            display_dataset_content(file_name, file_path, dataset_path)
            logging.debug(f"Displayed content for dataset: {dataset_path} in file: {file_name}")
        except Exception as e:
            st.error(f"Error processing selected dataset: {e}")
            logging.exception("Exception occurred while processing selected dataset.")

    # Display selected channels and provide options to remove them
    display_selected_channels()

    # Display combined data from selected channels
    display_combined_data()

    # Plotting Section
    plot_selected_channels()

    # Option to clear all selected channels
    if st.sidebar.button("Clear All Selected Channels"):
        clear_selected_channels()


if __name__ == "__main__":
    main()
