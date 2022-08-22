"""
Script for adding Patent-data into the memory
"""

from pathlib import Path
import glob
import json
import pandas as pd


class DOCPreparation:
    """Class for loading patent documents from multiple Jsonl files as pandas dataframe

    Arguments
    ---------
    path: path to the *.jsol files
    """
    def __init__(self, path=Path(__file__).parent / "../data"):
        self.path_to_patent_files = path
        self.all_files = glob.glob(str(self.path_to_patent_files) + "/*.jsonl")
        self._li = []

    def read_patents(self):
        """ Method for loading patent documents from multiple Jsonl files as a pandas dataframe
        """
        if self.all_files:
            for filename in self.all_files:
                try:
                    with open(filename, encoding="utf-8") as file:
                        sub_patent_df = pd.DataFrame(json.loads(line) for line in file)
                        self._li.append(sub_patent_df)
                except IOError:
                    print(filename, "File does not appear to exist.")

            patent_df = pd.concat(self._li, axis=0, ignore_index=True)
            return patent_df["contents"].to_frame(name="contents")
        print("File does not appear to exist.")
        return None

    def preprocess_patent(self, patent_data):
        """method for preprocessing patents data"""
