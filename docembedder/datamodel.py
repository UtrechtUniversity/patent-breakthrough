"""Data model for storing patent analysis data."""

from pathlib import Path
from typing import Union, Dict, List, Optional, Any, Tuple

import h5py
import numpy as np
from numpy import typing as npt
from scipy.sparse import csr_matrix

from docembedder.models.utils import create_model
from docembedder.models.base import AllEmbedType, BaseDocEmbedder


class DataModel():
    """Data model for and HDF5 file that keeps embeddings results.

    The groups are ordered as follows:

    embeddings: Stores all the embeddings
        {model}: Embeddings are sorted by model name first.
            {year}: Each window/year of embeddings has their own dataset.
                - (data, indices, indptr) for sparse embeddings, or:
                - (array) for dense embeddings
    year: Stores all patent numbers that were used for each window.
        {year}: One dataset for each window of all patent numbers used.
    cpc: Stores the correlations between CPC classifications of different patents
        {year}: They are stored by year/window
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
    def __init__(self, hdf5_file: Union[Path, str], read_only: bool=False):
        self.hdf5_file = hdf5_file
        if not read_only:
            self.handle = h5py.File(hdf5_file, "a")
            if "embeddings" not in self.handle:
                self.handle.create_group("embeddings")
                self.handle.create_group("year")
                self.handle.create_group("models")
                self.handle.create_group("cpc")
        else:
            self.handle = h5py.File(hdf5_file, "r")

    def store_year(self,
                   year: str,
                   test_id: npt.NDArray[np.int_],
                   train_id: npt.NDArray[np.int_]):
        """Store the patent numbers used for a window or check if they're the same.

        Arguments
        ---------
        year:
            Year or window of years used for the embeddings.
        test_id:
            Patent numbers for the embedding vectors.
        train_id:
            Patent numbers that were used to train the embedding vectors.
            This is usually the same as the test_id.
        """

        # Create the new window/year group
        try:
            year_group = self.handle[f"/year/{year}"]
        except KeyError:
            year_group = self.handle.create_group(f"/year/{year}")

        # Add the patent numbers for the test_ids
        if "test_id" in year_group:
            assert np.all(test_id == year_group["test_id"][...])
        else:
            year_group.create_dataset("test_id", data=test_id)

        # Add the patent numbers for the train_ids
        if "train_id" in year_group:
            assert np.all(train_id == year_group["train_id"][...])
        else:
            year_group.create_dataset("train_id", data=train_id)

    def store_embeddings(self,
                         year: str,
                         model_name: str,
                         embeddings: AllEmbedType,
                         overwrite: bool=False):
        """Store embeddings for a window/year.

        Arguments
        ---------
        year:
            Year or window name.
        model_name:
            Name of the model that generated the embeddings.
        overwrite:
            If True, overwrite embeddings if they exist.
        """
        dataset_group_str = f"/embeddings/{model_name}/{year}"
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
        else:
            raise ValueError(f"Not implemented datatype {type(embeddings)}")

    def store_cpc_correlations(self, year: str, data: Dict[str, npt.NDArray]):
        """Store CPC correlations for a year/window.

        Arguments
        ---------
        year:
            Window or year of the CPC correlations
        data:
            Correlations, as a dict with i_patents, j_patents, correlations.
        """
        if f"/cpc/{year}" in self.handle:
            return

        grp = self.handle.require_group(f"/cpc/{year}")
        grp.create_dataset("i_patents", data=data["i_patents"])
        grp.create_dataset("j_patents", data=data["j_patents"])
        grp.create_dataset("correlations", data=data["correlations"])

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

    def load_model(self, model_name) -> BaseDocEmbedder:
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
        return create_model(**model_dict)

    @property
    def model_names(self) -> List["str"]:
        """Names of all stored models."""
        return list(self.handle["embeddings"].keys())

    def iterate_embeddings(self, return_cpc: bool=False,
                           model_names: Optional[Union[str, List[str]]]=None):
        """Iterate over all embeddings in the data file.

        Arguments
        ---------
        return_cpc:
            Whether to get the CPC correlations
        """
        if model_names is None:
            model_list = list(self.model_names)
        elif isinstance(model_names, str):
            model_list = [model_names]
        else:
            model_list = model_names

        cpc_base_group = self.handle["/cpc"]

        # Iterate over the available windows.
        for year_str in self.handle["year"].keys():
            if return_cpc and year_str not in cpc_base_group:
                continue

            cur_results: Dict[str, Any] = {"embeddings": {}}
            for cur_model in model_list:
                model_group = self.handle[f"/embeddings/{cur_model}"]
                cur_results["embeddings"][cur_model] = (
                    self._get_embeddings(model_group[year_str]))
            if return_cpc:
                cur_results["cpc"] = {
                    "i_patents": cpc_base_group[year_str+"/i_patents"][...],
                    "j_patents": cpc_base_group[year_str+"/j_patents"][...],
                    "correlations": cpc_base_group[year_str+"/correlations"][...]
                }
            cur_results["patent_id"] = self.handle[f"/year/{year_str}/test_id"][...]
            cur_results["year"] = year_str
            yield cur_results

    def has_run(self, model_name: str, year: str) -> bool:
        """Compute whether a model has run on a certain window/year.

        Arguments
        ---------
        model_name:
            Name of the model.
        year:
            Window of year for the embedding.
        """
        return f"/embeddings/{model_name}/{year}" in self.handle

    def has_cpc(self, year: str) -> bool:
        """Compute whether there is CPC correlation data.

        Arguments
        ---------
        year:
            Window or year of the CPC.
        """
        return f"/cpc/{year}" in self.handle

    def has_year(self, year: str) -> bool:
        """Compute whether there is already an entry for a window/year.

        Arguments
        ---------
        year:
            Window or year.
        """
        return str(year) in self.handle["year"]

    def has_model(self, model_name: str) -> bool:
        """Compute whether there is already an entry for a model.

        Arguments
        ---------
        model_name:
            Name of the model (not to be confused with the model class).
        """
        return model_name in self.handle["models"]

    def __str__(self):
        ret_str = "Models:\n\n"
        for model_name in self.handle["/embeddings"].keys():
            ret_str += model_name + "\n"

        ret_str += "Years:\n\n"
        for year in self.handle["/year"].keys():
            ret_str += year + "\n"
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
            for year in model_group:
                year_group = model_group[year]
                if test_stale(year_group):
                    del self.handle[f"/embeddings/{model_name}/{year}"]

    def _get_embeddings(self, dataset_group):
        """Get the embeddings from a dataset in the data file."""
        if dataset_group.attrs["dtype"] == "array":
            return dataset_group["array"][...]
        if dataset_group.attrs["dtype"] == "csr_matrix":
            shape = dataset_group.attrs["shape"]
            data = dataset_group["data"][...]
            indices = dataset_group["indices"][...]
            indptr = dataset_group["indptr"][...]
            return csr_matrix((data, indices, indptr), shape=shape)
        raise ValueError(f"Unrecognized datatype {dataset_group.attr['dtype']}")

    def get_train_test_id(self, year: str) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Get the training and test patent numbers for a window/year.

        Arguments
        ---------
        year:
            Window or year.
        """
        year_group = self.handle[f"/year/{year}"]
        return year_group["train_id"][...], year_group["test_id"][...]

    def add_data(self, data_fp: Union[Path, str], delete_copy: bool=False):
        """Collect data from another file and add it.

        Arguments
        ---------
        data_fp:
            Other datafile to collect from.
        delete_copy:
            If True, delete the file that the data was collected from after collection.
        """
        with self.__class__(data_fp, read_only=False) as other:
            new_models = list(set(other.model_names) - set(self.model_names))
            for cur_model in new_models:
                self.handle["/embeddings"].copy(other.handle[f"/embeddings/{cur_model}"], cur_model)
            for year in other.handle["/year"].keys():
                if year not in self.handle["/year"]:
                    self.handle["/year"].copy(other.handle[f"/year/{year}"], year)
            for model_name in other.model_names:
                for year in other.handle[f"/embeddings/{model_name}"].keys():
                    group_name = f"/embeddings/{model_name}/{year}"
                    if group_name not in self.handle:
                        self.handle[f"/embeddings/{model_name}"].copy(
                            other.handle[group_name],
                            year
                        )
                if year in other.handle["/cpc"].keys() and year not in self.handle["/cpc"].keys():
                    self.handle["/cpc"].copy(other.handle[f"/cpc/{year}"], year)
        if delete_copy:
            Path(data_fp).unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_detect_stale()
        self.handle.close()
        return exc_type is None
