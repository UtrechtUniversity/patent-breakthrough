import h5py
import numpy as np

from docembedder.classification import PatentClassification
from scipy.sparse import csr_matrix
from docembedder.models.utils import create_model


class DataModel():
    def __init__(self, hdf5_file, read_only=False):
        self.hdf5_file = hdf5_file
        if not read_only:
            self.handle = h5py.File(hdf5_file, "a")
            if "embeddings" not in self.handle:
                self.handle.create_group("embeddings")
                self.handle.create_group("year")
                self.handle.create_group("models")
        else:
            self.handle = h5py.File(hdf5_file, "r")

    def compute_embeddings(self, model, model_name, train_patents, test_patents, year,
                           overwrite=False):
        dataset_group_str = f"/embeddings/{model_name}/{year}"
        if dataset_group_str in self.handle and overwrite:
            del self.handle[dataset_group_str]
        elif dataset_group_str in self.handle:
            return
        dataset_group = self.handle.create_group(dataset_group_str)
        train_id = [pat["patent"] for pat in train_patents]
        test_id = [pat["patent"] for pat in test_patents]
        self._add_year(year, test_id, train_id)

        model.fit([pat["contents"] for pat in train_patents])
        embeddings = model.transform([pat["contents"] for pat in test_patents])
        self._store_embeddings(dataset_group, embeddings)
        model_group = self.handle[f"/embeddings/{model_name}"]
        for key, value in model.settings.items():
            model_group.attrs[key] = value

    def store_year(self, year, test_id, train_id):
            # Set training data
        try:
            year_group = self.handle[f"/year/{year}"]
        except KeyError:
            year_group = self.handle.create_group(f"/year/{year}")

        if "test_id" in year_group:
            assert np.all(test_id == year_group["test_id"][...])
        else:
            year_group.create_dataset("test_id", data=test_id)

        if "train_id" in year_group:
            assert np.all(train_id == year_group["train_id"][...])
        else:
            year_group.create_dataset("train_id", data=train_id)

    def store_embeddings(self, model_name, embeddings, year, overwrite=False):
        dataset_group_str = f"/embeddings/{model_name}/{year}"
        if dataset_group_str in self.handle and overwrite:
            del self.handle[dataset_group_str]
        elif dataset_group_str in self.handle:
            return
        dataset_group = self.handle.create_group(dataset_group_str)
        # train_id = [pat["patent"] for pat in train_patents]
        # test_id = [pat["patent"] for pat in test_patents]
        # self._add_year(year, test_id, train_id)

        # model.fit([pat["contents"] for pat in train_patents])
        # embeddings = model.transform([pat["contents"] for pat in test_patents])
        self._store_embeddings(dataset_group, embeddings)

    def sample_cpc_correlations(self, class_fp, year, samples_per_patent=None):
        cpc = PatentClassification(class_fp)
        try:
            pat_id = self.handle[f"/year/{year}/test_id"][...]
        except KeyError as exc:
            raise ValueError("First run a model on the patents before classification.") from exc

        i_patents, j_patents, correlations = cpc.sample_cpc_correlations(
            pat_id, samples_per_patent)

        if f"/cpc/{year}" in self.handle:
            return

        grp = self.handle.require_group(f"/cpc/{year}")
        grp.create_dataset("i_patents", data=i_patents)
        grp.create_dataset("j_patents", data=j_patents)
        grp.create_dataset("correlations", data=correlations)

    def store_cpc_correlations(self, data, year):
        # if "test_"
        # cpc = PatentClassification(class_fp)
        # try:
            # pat_id = self.handle[f"/year/{year}/test_id"][...]
        # except KeyError as exc:
            # raise ValueError("First run a model on the patents before classification.") from exc

        # i_patents, j_patents, correlations = cpc.sample_cpc_correlations(
            # pat_id, samples_per_patent)

        if f"/cpc/{year}" in self.handle:
            return

        grp = self.handle.require_group(f"/cpc/{year}")
        grp.create_dataset("i_patents", data=data["i_patents"])
        grp.create_dataset("j_patents", data=data["j_patents"])
        grp.create_dataset("correlations", data=data["correlations"])

    def store_model(self, model_name, model):
        model_group = self.handle.create_group(f"/models/{model_name}")
        for key, value in model.settings.items():
            model_group.attrs[key] = value
        model_group.attrs["model_type"] = model.__class__.__name__

    def load_model(self, model_name):
        model_dict = dict(self.handle[f"/models/{model_name}"].attrs)
        return create_model(**model_dict)

    @property
    def model_names(self):
        return list(self.handle["embeddings"].keys())

    def iterate_embeddings(self, return_cpc=False, model_names=None):
        if model_names is None:
            model_list = list(self.model_names)
        elif isinstance(model_names, str):
            model_list = [model_names]
        else:
            model_list = model_names

        cpc_base_group = self.handle["/cpc"]

        for year_str in self.handle["year"].keys():
            if return_cpc and year_str not in cpc_base_group:
                continue

            cur_results = {"embeddings": {}}
            for cur_model in model_list:
                model_group = self.handle[f"/embeddings/{cur_model}"]
                assert cur_model != "cpc", "Please give your model another name than cpc."
                cur_results["embeddings"][cur_model] = (
                    self._get_embeddings(model_group[year_str]))
            if return_cpc:
                cur_results["cpc"] = {
                    "i_patents": cpc_base_group[year_str+"/i_patents"][...],
                    "j_patents": cpc_base_group[year_str+"/j_patents"][...],
                    "correlations": cpc_base_group[year_str+"/correlations"][...]
                }
            cur_results["patent_id"] = self.handle[f"/year/{year_str}/test_id"][...],
            cur_results["year"] = int(year_str)
            yield cur_results

    def has_run(self, model_name, year):
        return f"/embeddings/{model_name}/{year}" in self.handle

    def has_cpc(self, year):
        return f"/cpc/{year}" in self.handle

    def has_year(self, year):
        return str(year) in self.handle["year"]

    def has_model(self, model_name):
        return model_name in self.handle["models"]

    def __str__(self):
        ret_str = "Models:\n\n"
        for model_name in self.handle[f"/embeddings"].keys():
            ret_str += model_name + "\n"

        ret_str += "Years:\n\n"
        for year in self.handle[f"/year"].keys():
            ret_str += year + "\n"
        return ret_str

    def _remove_detect_stale(self):
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

    def _store_embeddings(self, dataset_group, embeddings):
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

    def _get_embeddings(self, dataset_group):
        if dataset_group.attrs["dtype"] == "array":
            return dataset_group["array"][...]
        elif dataset_group.attrs["dtype"] == "csr_matrix":
            shape = dataset_group.attrs["shape"]
            data = dataset_group["data"][...]
            indices = dataset_group["indices"][...]
            indptr = dataset_group["indptr"][...]
            return csr_matrix((data, indices, indptr), shape=shape)
        raise ValueError(f"Unrecognized datatype {dataset_group.attr['dtype']}")

    def get_train_test_id(self, year):
        year_group = self.handle[f"/year/{year}"]
        return year_group["train_id"][...], year_group["test_id"][...]

    def merge_with(self, data_fp, delete_copy=False):
        with self.__class__(data_file, read_only=False) as other:
            new_models = list(set(other.model_names) - set(self.model_names))
            exist_models = list(set(self.model_names) & set(other.model_names))
            for cur_model in exist_models:
                self.handle["/embeddings"].copy(other.handle[f"/embeddings/{cur_model}"], cur_model)
                
            
            for year in other.handle["/year"].keys():
                if year in self.handle["/year"]:
                    continue
                self.handle["/year"].copy(other.handle[f"/year/{year}"], year)
                for cur_model in exist_models:
                    self.handle[f"/embeddings/{model_name}"].copy(
                        other.handle[f"/embeddings/{cur_model}/{year}"],
                        year)
                if year in other.handle["/cpc"].keys() and year not in self.handle["/cpc"].keys():
                    self.handle["/cpc"].copy(other.handle[f"/cpc/{year}"], year)
                    

    def _add_year(self, year, test_id, train_id):
        # Set training data
        try:
            year_group = self.handle[f"/year/{year}"]
        except KeyError:
            year_group = self.handle.create_group(f"/year/{year}")

        if "test_id" in year_group:
            assert np.all(test_id == year_group["test_id"][...])
        else:
            year_group.create_dataset("test_id", data=test_id)

        if "train_id" in year_group:
            assert np.all(train_id == year_group["train_id"][...])
        else:
            year_group.create_dataset("train_id", data=train_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._remove_detect_stale()
        self.handle.close()
        return exc_type is None
