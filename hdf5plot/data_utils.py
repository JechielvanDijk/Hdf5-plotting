import logging

import h5py
import numpy as np
import pandas as pd
import streamlit as st
import scipy.constants as cs
from lmfit.models import LinearModel


def format_numbers(val):
    if pd.isnull(val):
        return ""
    if isinstance(val, (int, float)):
        if val == 0:
            return "0.00"
        return f"{val:.4g}"
    return val


def decode_if_bytes(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return x


def parse_hdf5_structure(file):
    structure = {}
    logging.debug("Starting to parse HDF5 file structure.")

    def visit(name, obj):
        logging.debug(f"Visiting: {name}")
        path_parts = name.split("/")
        current = structure
        for part in path_parts:
            if part:
                current = current.setdefault(part, {})
        if isinstance(obj, h5py.Dataset):
            current["_type"] = "Dataset"
            current["_shape"] = obj.shape
            current["_dtype"] = str(obj.dtype)
            logging.debug(
                f"Added Dataset: {name} | Shape: {obj.shape} | Dtype: {obj.dtype}"
            )
        elif isinstance(obj, h5py.Group):
            current["_type"] = "Group"
            logging.debug(f"Added Group: {name}")

    file.visititems(visit)
    logging.debug("Completed parsing HDF5 file structure.")
    return structure


def load_dataset(file_path, dataset_path):
    logging.debug(f"Loading dataset from file: {file_path} | Dataset path: {dataset_path}")
    try:
        with h5py.File(file_path, "r") as f:
            dataset = f[dataset_path][()]
        logging.debug(
            f"Loaded dataset | Shape: {dataset.shape}, Dtype: {dataset.dtype}"
        )
        return dataset
    except KeyError:
        st.error(f"Dataset path '{dataset_path}' not found in the HDF5 file.")
        logging.error(f"Dataset path '{dataset_path}' not found in the HDF5 file.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        logging.exception("Exception occurred while loading dataset.")
        return None


def extract_density(field, rxy, field_cutoffs):
    if len(field.shape) >= 2:
        input_is_1d = False
        original_shape = field.shape[:-1]
        trace_number = np.prod(original_shape)
        field = field.reshape((trace_number, -1))
        rxy = rxy.reshape((trace_number, -1))
        if len(field_cutoffs) == 2:
            fc = np.empty(original_shape + (2,))
            fc[..., 0] = field_cutoffs[0]
            fc[..., 1] = field_cutoffs[1]
            field_cutoffs = fc
        field_cutoffs = field_cutoffs.reshape((trace_number, -1))
    else:
        input_is_1d = True
        trace_number = 1
        field = np.array((field,))
        rxy = np.array((rxy,))
        field_cutoffs = np.array((field_cutoffs,))

    results = np.empty((2, trace_number))
    fits = []
    model = LinearModel()

    for i in range(trace_number):
        mask = ~np.isnan(field[i])
        f = field[i][mask]
        r = rxy[i][mask]
        start_field, stop_field = field_cutoffs[i]
        field_mask = (start_field <= f) & (f <= stop_field)
        f = f[field_mask]
        r = r[field_mask]
        res = model.fit(r, x=f)
        results[0, i] = 1 / res.best_values["slope"] / cs.e
        results[1, i] = results[0, i] * (
            res.params["slope"].stderr / res.best_values["slope"]
        )
        fits.append(res)

    if input_is_1d:
        return (*results[:, 0], fits[0])
    return (*results.reshape((2,) + original_shape), np.reshape(fits, original_shape))


def extract_mobility(field, rxx, ryy, density, geometric_factor):
    if len(field.shape) >= 2:
        input_is_1d = False
        original_shape = field.shape[:-1]
        trace_number = np.prod(original_shape)
        field = field.reshape((trace_number, -1))
        rxx = rxx.reshape((trace_number, -1))
        ryy = ryy.reshape((trace_number, -1))
    else:
        input_is_1d = True
        trace_number = 1
        field = np.array((field,))
        rxx = np.array((rxx,))
        ryy = np.array((ryy,))

    r0 = np.empty((2, trace_number))
    for i in range(trace_number):
        min_field_ind = np.argmin(np.abs(field[i]))
        r0[0, i] = rxx[i, min_field_ind]
        r0[1, i] = ryy[i, min_field_ind]

    r0 *= geometric_factor
    mob = 1 / cs.e / density / r0

    if input_is_1d:
        return mob[:, 0]
    return mob.reshape((2,) + original_shape)
