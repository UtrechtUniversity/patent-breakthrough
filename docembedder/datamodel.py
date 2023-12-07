"""Data model for storing patent analysis data."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import io

import h5py
import numpy as np
from numpy import typing as npt
from scipy.sparse import csr_matrix

from docembedder.models.utils import create_model, create_preprocessor
from docembedder.models.base import AllEmbedType, BaseDocEmbedder
from docembedder.preprocessor.preprocessor import Preprocessor
from docembedder.typing import FileType


class DataModel():  # pylint: disable=too-many-public-methods
    """Data model for and HDF5 file that keeps embeddings results.

    The groups are ordered as follows:

    embeddings: Stores all the embeddings
        {model}: Embeddings are sorted by model name first.
            {window}: Each window/year of embeddings has their own dataset.
                - (data, indices, indptr) for sparse embeddings, or:
                - (array) for dense embeddings
    impacts_novelties: Stores all impact/novelty values
        {model}: Model name
            {window}: window name is the dataset.
    windows: Stores all patent numbers that were used for each window.
        {window}: One dataset for each window of all patent numbers used.
    cpc: Stores the correlations between CPC classifications of different patents
        {window}: They are stored by year/window
            i_patents: Array of patent indices i (not to be confused with patent numbers)
            j_patents: Array of patent indices j
            correlation: Array with correlation between i and j.
    models: Stores the models that were run.
        {model}: Group with attributes to recreate a model.

    Arguments
    ---------
    hdf5_file:
        file that stores the embeddings.
    read_only:
        Open the file in read only mode, no writes possible.
    """
    def __init__(self, hdf5_file: FileType, read_only: bool=False):
        self.hdf5_file = hdf5_file
        self.read_only = read_only
        if not read_only:
            self.handle = h5py.File(hdf5_file, "a")
            if "embeddings" not in self.handle:
                self.handle.create_group("embeddings")
                self.handle.create_group("windows")
                self.handle.create_group("models")
                self.handle.create_group("preprocessors")
                self.handle.create_group("cpc")
                self.handle.create_group("impact_novelty")
                self.handle.attrs["docembedder-version"] = "unknown"  # Should be fixed in case
        else:
            self.handle = h5py.File(hdf5_file, "r")

    def store_window(self,
                     window_name: str,
                     patent_id: npt.NDArray[np.int_],
                     year: npt.NDArray[np.int_]
                     ):
        """Store the patent numbers used for a window or check if they're the same.

        Arguments
        ---------
        window_name:
            Year or window of years used for the embeddings.
        patent_id:
            Patent numbers for the embedding vectors.
        year:
            Year of each of the patents in the window.
        """

        if len(patent_id) != len(year):
            raise ValueError("Cannot store window with patent_id of different length"
                             " than year.")

        # Create the new window/year group
        try:
            window_group = self.handle[f"/windows/{window_name}"]
        except KeyError:
            window_group = self.handle.create_group(f"/windows/{window_name}")

        # Add the patent numbers for the patents in this window
        if "patent_id" in window_group:
            assert np.all(patent_id == window_group["patent_id"][...])
        else:
            window_group.create_dataset("patent_id", data=patent_id)

        # Add the year for each of the patents
        if "year" in window_group:
            assert np.all(year == window_group["year"][...])
        else:
            window_group.create_dataset("year", data=year)

    def load_window(self, window_name: str) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Load the patent numbers and year of issue.

        Arguments
        ---------
        window_name:
            Name of the window to load.

        Returns
        -------
        patent_id:
            Patent number for each of the patents in the window.
        year:
            Year of patent issue for each of the patents.
        """
        return (
            self.handle[f"/windows/{window_name}/patent_id"][...],
            self.handle[f"/windows/{window_name}/year"][...],
        )

    def store_embeddings(self,
                         window_name: str,
                         model_name: str,
                         embeddings: AllEmbedType,
                         overwrite: bool=False):
        """Store embeddings for a window/year.

        Arguments
        ---------
        window_name:
            Year or window name.
        model_name:
            Name of the model that generated the embeddings.
        overwrite:
            If True, overwrite embeddings if they exist.
        """
        if not isinstance(embeddings, (np.ndarray, csr_matrix)):
            raise ValueError(f"Not implemented datatype {type(embeddings)}")

        dataset_group_str = f"/embeddings/{model_name}/{window_name}"
        if dataset_group_str in self.handle and overwrite:
            del self.handle[dataset_group_str]
        elif dataset_group_str in self.handle:
            return
        dataset_group = self.handle.create_group(dataset_group_str)

        if isinstance(embeddings, np.ndarray):
            dataset_group.create_dataset("array", data=embeddings)
            dataset_group.attrs["dtype"] = "array"
        elif isinstance(embeddings, csr_matrix):
            dataset_group.create_dataset("data", data=embeddings.data)
            dataset_group.create_dataset("indices", data=embeddings.indices)
            dataset_group.create_dataset("indptr", data=embeddings.indptr)
            dataset_group.attrs["dtype"] = "csr_matrix"
            dataset_group.attrs["shape"] = embeddings.shape

    def load_embeddings(self, window_name: str, model_name: str) -> AllEmbedType:
        """Load embeddings for a window/year.

        Arguments
        ---------
        window_name:
            Year or window name.
        model_name:
            Name of the model that generated the embeddings.

        Returns
        -------
        embeddings:
            Embeddings for that window/model.
        """
        dataset_group = self.handle[f"/embeddings/{model_name}/{window_name}"]
        if dataset_group.attrs["dtype"] == "array":
            return dataset_group["array"][...]
        if dataset_group.attrs["dtype"] == "csr_matrix":
            shape = dataset_group.attrs["shape"]
            data = dataset_group["data"][...]
            indices = dataset_group["indices"][...]
            indptr = dataset_group["indptr"][...]
            return csr_matrix((data, indices, indptr), shape=shape)
        raise ValueError(f"Unrecognized datatype {dataset_group.attr['dtype']}")

    def store_cpc_correlations(self, window_name: str, data: dict[str, npt.NDArray]):
        """Store CPC correlations for a year/window.

        Arguments
        ---------
        window:
            Window or year of the CPC correlations
        data:
            Correlations, as a dict with i_patents, j_patents, correlations.
        """
        if f"/cpc/{window_name}" in self.handle:
            return

        grp = self.handle.require_group(f"/cpc/{window_name}")
        grp.create_dataset("i_patents", data=data["i_patents"])
        grp.create_dataset("j_patents", data=data["j_patents"])
        grp.create_dataset("correlations", data=data["correlations"])

    def load_cpc_correlations(self, window_name: str) -> dict[str, npt.NDArray]:
        """Store CPC correlations for a year/window.

        Arguments
        ---------
        window_name:
            Window or year of the CPC correlations

        Returns
        -------
        data:
            Correlations, as a dict with i_patents, j_patents, correlations.
        """
        cpc_group = self.handle[f"/cpc/{window_name}"]
        return {
            "i_patents": cpc_group["i_patents"][...],
            "j_patents": cpc_group["j_patents"][...],
            "correlations": cpc_group["correlations"][...]
        }

    def store_cpc_spearmanr(self, window_name: str, model_name: str, correlation: float):
        """Store CPC spearmanr results.

        Arguments
        ---------
        window_name:
            Name of the window/year to store.
        model_name:
            Name of the model to store the correlations.
        correlation:
            Value of the Spearman-R correlation.
        """
        self.handle[f"/embeddings/{model_name}/{window_name}"].attrs["cpc_spearmanr"] = correlation

    def load_cpc_spearmanr(self, window_name: str, model_name: str) -> float:
        """Load CPC spearmanr results.

        Arguments
        ---------
        window_name:
            Name of the window/year to store.
        model_name:
            Name of the model to store the correlations.
        correlation:
            Value of the Spearman-R correlation.
        """
        return self.handle[f"/embeddings/{model_name}/{window_name}"].attrs["cpc_spearmanr"]

    def store_model(self, model_name: str, model: BaseDocEmbedder):
        """Store the settings of a model, to be reinitialized

        Arguments
        ---------
        model_name:
            Name of the model to be stored (not the name of the class).
        model:
            Model instance to store.
        """
        model_group = self.handle.create_group(f"/models/{model_name}")
        for key, value in model.settings.items():
            model_group.attrs[key] = value
        model_group.attrs["model_type"] = model.__class__.__name__

    def load_model(self, model_name: str) -> BaseDocEmbedder:
        """Load model from the settings in the file.

        Arguments
        ---------
        model_name:
            Name of the model to load from file.

        Returns
        -------
        model:
            Newly initialized (untrained) model.
        """
        model_dict = dict(self.handle[f"/models/{model_name}"].attrs)
        model_type = model_dict.pop("model_type")
        return create_model(model_type, model_dict)

    def store_preprocessor(self, prep_name: str, preprocessor: Preprocessor):
        """Store preprocessor information."""
        prep_group = self.handle.create_group(f"/preprocessors/{prep_name}")
        for key, value in preprocessor.settings.items():
            prep_group.attrs[key] = value
        prep_group.attrs["prep_type"] = preprocessor.__class__.__name__

    def load_preprocessor(self, prep_name: str, **kwargs) -> Preprocessor:
        """Create preprocessor from stored information."""
        prep_dict = dict(self.handle[f"/preprocessors/{prep_name}"].attrs)
        prep_type = prep_dict.pop("prep_type")
        prep_dict.update(kwargs)
        prep = create_preprocessor(prep_type, prep_dict)
        return prep

    def store_impact_novelty(self, window_name: str, model_name: str,
                             data: dict,
                             overwrite: bool = False):
        """Store impact and novelty for a window/year.

        Arguments
        ---------
        window_name:
            Year or window name.
        model_name:
            Name of the model that generated the embeddings.
        impacts:
            An array containing the impacts per model/window.
        exponent:
            Exponent used to compute the impact.
        overwrite:
            If True, overwrite embeddings if they exist.
        """
        exponent = data["exponent"]

        dataset_group_str = f"/impact_novelty/{model_name}/{window_name}/{exponent}"
        if dataset_group_str in self.handle and overwrite:
            del self.handle[dataset_group_str]
        elif dataset_group_str in self.handle:
            return
        dataset_group = self.handle.create_group(dataset_group_str)

        dataset_group.attrs["focal_year"] = data["focal_year"]
        dataset_group.create_dataset("patent_ids", data=data["patent_ids"])
        dataset_group.create_dataset("novelty", data=data["novelty"])
        dataset_group.create_dataset("impact", data=data["impact"])

    def load_impact_novelty(self, window_name: str, model_name: str, exponent: float
                            ) -> dict:
        """Load impacts for a window/year.

        Arguments
        ---------
        window_name:
            Year or window name.
        model_name:
            Name of the model.

        Returns
        -------
        Impacts:
            list of impacts for that window/model.
        """
        dataset_group = self.handle[f"/impact_novelty/{model_name}/{window_name}/{exponent}"]
        results = {
            "impact": dataset_group["impact"][...],
            "novelty": dataset_group["novelty"][...],
            "focal_year": dataset_group.attrs["focal_year"],
            "patent_ids": dataset_group["patent_ids"][...],
            "exponent": exponent,
        }
        return results

    @property
    def model_names(self) -> list[str]:
        """Names of all stored models."""
        return list(self.handle["embeddings"].keys())

    @property
    def window_list(self) -> list[str]:
        """Names of all stored models."""
        return list(self.handle["windows"].keys())

    def iterate_window_models(self,
                              window_name: Optional[str] = None,
                              model_name: Optional[str] = None) -> Iterable[tuple[str, str]]:
        """Iterate over all available windows/models.

        Returns
        -------
        window, model_name:
            Window and model_name for each combination that has an embedding.
        """
        for cur_window_name in self.handle["windows"]:
            if window_name is not None and cur_window_name != window_name:
                continue
            for cur_model_name in self.model_names:
                if model_name is not None and cur_model_name != model_name:
                    continue
                if f"/embeddings/{cur_model_name}/{cur_window_name}" not in self.handle:
                    continue
                yield cur_window_name, cur_model_name

    def has_run(self, prep_name: str, embed_name: str, window_name: str) -> bool:
        """Compute whether a model has run on a certain window/year.

        Arguments
        ---------
        prep_name:
            Name of the preprocessor.
        embed_name:
            Name of the embedding model.
        window_name:
            Window of year for the embedding.
        """
        return f"/embeddings/{prep_name}-{embed_name}/{window_name}" in self.handle

    def has_cpc(self, window_name: str) -> bool:
        """Compute whether there is CPC correlation data.

        Arguments
        ---------
        window_name:
            Window or year of the CPC.
        """
        return f"/cpc/{window_name}" in self.handle

    def has_window(self, window_name: str) -> bool:
        """Compute whether there is already an entry for a window/year.

        Arguments
        ---------
        window_name:
            Window or year.
        """
        return str(window_name) in self.handle["windows"]

    def has_model(self, model_name: str) -> bool:
        """Compute whether there is already an entry for a model.

        Arguments
        ---------
        model_name:
            Name of the model (not to be confused with the model class).
        """
        return model_name in self.handle["models"]

    def has_prep(self, prep_name: str) -> bool:
        """Return whether a preprocessor exists with that name."""
        return prep_name in self.handle["preprocessors"]

    def __str__(self):
        ret_str = "Models:\n\n"
        for model_name in self.handle["/embeddings"].keys():
            ret_str += model_name + "\n"

        ret_str += "Windows:\n\n"
        for window_name in self.handle["/windows"].keys():
            ret_str += window_name + "\n"
        return ret_str

    def _remove_detect_stale(self):
        """Remove dead groups/datasets from the data file to stop corruption."""
        def test_stale(group):
            if "dtype" not in group.attrs:
                return True
            if group.attrs["dtype"] == "array":
                if "array" in group:
                    return False
            elif group.attrs["dtype"] == "csr_matrix":
                if ("data" in group and
                        "indices" in group and
                        "indptr" in group):
                    return False
            return True

        for model_name in self.handle["/embeddings"]:
            model_group = self.handle[f"/embeddings/{model_name}"]
            for window_name in model_group:
                window_group = model_group[window_name]
                if test_stale(window_group):
                    del self.handle[f"/embeddings/{model_name}/{window_name}"]

    def add_data(self, data_fp: FileType, delete_copy: bool=False):
        """Collect data from another file and add it.

        Arguments
        ---------
        data_fp:
            Other datafile to collect from.
        delete_copy:
            If True, delete the file that the data was collected from after collection.
        """
        if not (isinstance(data_fp, io.BytesIO) or Path(data_fp).is_file()):
            raise FileNotFoundError(f"Cannot find file {data_fp} to add to datamodel.")
        with self.__class__(data_fp, read_only=False) as other:
            new_models = list(set(other.model_names) - set(self.model_names))
            for cur_model in new_models:
                self.handle["/embeddings"].copy(other.handle[f"/embeddings/{cur_model}"], cur_model)
            for window_name in other.handle["/windows"].keys():
                if window_name not in self.handle["/windows"]:
                    self.handle["/windows"].copy(other.handle[f"/windows/{window_name}"],
                                                 window_name)
            for model_name in other.model_names:
                for window_name in other.handle[f"/embeddings/{model_name}"].keys():
                    group_name = f"/embeddings/{model_name}/{window_name}"
                    if group_name not in self.handle:
                        self.handle[f"/embeddings/{model_name}"].copy(
                            other.handle[group_name],
                            window_name
                        )
                if (window_name in other.handle["/cpc"].keys()
                        and window_name not in self.handle["/cpc"].keys()):
                    self.handle["/cpc"].copy(other.handle[f"/cpc/{window_name}"], window_name)
        if delete_copy and not isinstance(data_fp, io.BytesIO):
            Path(data_fp).unlink()

    def remove_model(self, model_name: str):
        """Remove all data and information about a model from the data

        This will attempt to remove the serialization of the model itself,
        the impacts and novelties and the embeddings. If it can't find any
        of these items it will throw a KeyError.

        Parameters
        ----------
        model_name:
            Name/tag of the model (combined prep and classifier).
        """
        n_remove = 0
        # Remove model name/settings.
        try:
            del self.handle[f"/models/{model_name}"]
            n_remove += 1
        except KeyError:
            pass

        # Remove impacts/novelties
        try:
            del self.handle[f"/impact_novelty/{model_name}"]
            n_remove += 1
        except KeyError:
            pass

        # Remove embeddings
        try:
            del self.handle[f"/embeddings/{model_name}"]
            n_remove += 1
        except KeyError:
            pass

        if n_remove == 0:
            raise KeyError(f"Could not find model with name '{model_name}' in the data.")

    def __enter__(self) -> DataModel:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_detect_stale()
        self.handle.flush()
        if not isinstance(self.hdf5_file, io.BytesIO):
            self.handle.close()
        return exc_type is None
