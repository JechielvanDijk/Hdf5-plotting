import json
import logging
import os
import shutil
from datetime import datetime

import streamlit as st

SESSIONS_FOLDER = "sessions"


def save_session():
    """Save the current Streamlit session."""
    if not os.path.exists(SESSIONS_FOLDER):
        os.makedirs(SESSIONS_FOLDER)

    custom_name = st.session_state.get("custom_session_name", "").strip()
    base_name = custom_name or st.session_state.get("main_plot_title", "session")

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{base_name}_{dt_str}"
    session_dir = os.path.join(SESSIONS_FOLDER, session_name)
    os.makedirs(session_dir, exist_ok=True)

    session_config = {}
    for key, value in st.session_state.items():
        if key.startswith("remove_"):
            continue
        try:
            json.dumps(value)
            session_config[key] = value
        except (TypeError, OverflowError):
            continue

    config_path = os.path.join(session_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(session_config, f)

    files_dir = os.path.join(session_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    if "temp_files" in st.session_state:
        for file_name, file_path in st.session_state["temp_files"].items():
            dest_path = os.path.join(files_dir, file_name)
            shutil.copy(file_path, dest_path)

    st.session_state["current_session_name"] = session_name
    st.success(f"Session saved as '{session_name}'.")


def load_session(session_name):
    """Load a previously saved session."""
    session_dir = os.path.join(SESSIONS_FOLDER, session_name)
    config_path = os.path.join(session_dir, "config.json")
    if not os.path.exists(config_path):
        st.error("Session configuration not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    st.session_state.clear()

    skip_keys = [
        "load_session_select",
        "save_session_button",
        "load_session_button",
        "update_session_button",
        "clear_channels_button",
        "generate_plot",
        "add_selected_channels",
    ]

    for key, value in config.items():
        if key in skip_keys or key.startswith("remove_") or key.startswith("add_selected_channels"):
            continue
        st.session_state[key] = value

    files_dir = os.path.join(session_dir, "files")
    if os.path.exists(files_dir):
        files = {}
        for file_name in os.listdir(files_dir):
            files[file_name] = os.path.join(files_dir, file_name)
        st.session_state["temp_files"] = files
        logging.debug(f"Restored uploaded files from {files_dir}")

    st.session_state["current_session_name"] = session_name
    st.success(f"Session '{session_name}' loaded.")


def update_session(session_name):
    """Overwrite an existing saved session."""
    session_dir = os.path.join(SESSIONS_FOLDER, session_name)
    if not os.path.exists(session_dir):
        st.error("Session folder not found.")
        return

    session_config = {}
    for key, value in st.session_state.items():
        try:
            json.dumps(value)
            session_config[key] = value
        except (TypeError, OverflowError):
            continue

    config_path = os.path.join(session_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(session_config, f)
    logging.info(f"Session configuration updated at {config_path}")

    files_dir = os.path.join(session_dir, "files")
    os.makedirs(files_dir, exist_ok=True)
    if "temp_files" in st.session_state:
        for file_name, file_path in st.session_state["temp_files"].items():
            dest_path = os.path.join(files_dir, file_name)
            if os.path.abspath(file_path) == os.path.abspath(dest_path):
                continue
            shutil.copy(file_path, dest_path)
            logging.debug(f"Updated file {file_name} to {dest_path}")

    st.success(f"Session '{session_name}' updated.")


def add_selected_channels(selected_channels):
    st.session_state["selected_channels"].extend(selected_channels)
    logging.debug(f"Added channels to session state: {selected_channels}")


def remove_selected_channel(index):
    try:
        channel = st.session_state["selected_channels"].pop(index)
        logging.debug(f"Removed channel from session state: {channel}")
        file_path = channel["file_path"]
        still_used = any(ch["file_path"] == file_path for ch in st.session_state["selected_channels"])
        if not still_used:
            try:
                os.unlink(file_path)
                file_name = channel["file_name"]
                if file_name in st.session_state["temp_files"]:
                    del st.session_state["temp_files"][file_name]
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
    for channel in st.session_state["selected_channels"]:
        file_path = channel["file_path"]
        try:
            os.unlink(file_path)
            logging.debug(f"Deleted temporary file: {file_path}")
        except FileNotFoundError:
            logging.warning(f"Temporary file already deleted: {file_path}")
        except Exception as e:
            st.error(f"Error deleting temporary file {file_path}: {e}")
            logging.error(f"Error deleting temporary file {file_path}: {e}")
    st.session_state["selected_channels"] = []
    st.session_state["temp_files"] = {}
    st.success("All selected channels have been cleared and temporary files deleted.")
    logging.debug("Cleared all selected channels and deleted temporary files.")


def initialize_session_state():
    if "selected_dataset" not in st.session_state:
        st.session_state["selected_dataset"] = None
    if "selected_channels" not in st.session_state:
        st.session_state["selected_channels"] = []
    if "temp_files" not in st.session_state:
        st.session_state["temp_files"] = {}
    if "plots_history" not in st.session_state:
        st.session_state["plots_history"] = {}
