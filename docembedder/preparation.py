"""
Script for adding Patent-data into the memory
"""
import glob
import json

import pandas as pd


class DOCPreparation:
    """Class for loading patent documents from multiple Jsonl files as pandas dataframe

    """
    def __init__(self):
        self.all_files = glob.glob("./data/patents_cleaned_complete" + "/*.jsonl")
        if not self.all_files:
            print("File does not appear to exist.")
        self._li = []
        self.columns = ['patent', 'contents', 'year']

    def read_patents(self):
        """
        Method for loading patent documents from multiple Jsonl files and
        concatenate them as a pandas dataframe
        """
        if self.all_files:
            for filename in self.all_files:
                with open(filename, encoding="utf-8") as file:
                    sub_patent_df = pd.DataFrame(json.loads(line) for line in file)
                    self._li.append(sub_patent_df)

            patent_df = pd.concat(self._li, axis=0, ignore_index=True)

            patent_df[self.columns].to_csv('./data/patents_concatenated.csv', index=False)
            return patent_df[self.columns]

        return None

    def preprocess_patent(self, patent_df):
        """Method for preprocessing patents data"""
        pass
