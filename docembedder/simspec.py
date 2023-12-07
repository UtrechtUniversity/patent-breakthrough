"""General utilities for performing runs."""

from __future__ import annotations
from typing import Optional, Iterable

from docembedder.datamodel import DataModel
from docembedder.typing import FileType


STARTING_YEAR = 1838  # First year of the patents dataset


class SimulationSpecification():
    """Specification for doing runs.

    It uses the starting year so that windows start regularly and always
    from the same years, independent of where the run itself starts. All years
    between the year_start and year_end will be present in at least one of the windows.
    It is possible that some years will be included that are outside this interval.

    Arguments
    ---------
    year_start:
        Start of the windows to run the models on.
    year_end:
        End of the windows to run the models on. This year is not included, so if end year is 1902,
        then it the last year that is forced to run would be 1901.
    window_size:
        Number of years in each window.
    window_shift:
        Shift between consecutive windows. If None, each consecutive window is shifted by
        the window_size divided by 2 rounded up.
    cpc_samples_per_patent:
        Number of CPC correlation samples per patent.
    debug_max_patents:
        Only read the first x patents from the file to speed up computation.
        Leave at None for not skipping anything.
    n_patents_per_window:
        Number of patents to be drawn for each window. If None, all patents
        are used.
    """
    def __init__(self,  # pylint: disable=too-many-arguments
                 year_start: int,
                 year_end: int,
                 window_size: int=1,
                 window_shift: Optional[int]=None,
                 cpc_samples_per_patent: int=10,
                 debug_max_patents: Optional[int]=None,
                 n_patents_per_window: Optional[int]=None):
        self.year_start = year_start
        self.year_end = year_end
        self.window_size = window_size
        if window_shift is None:
            self.window_shift = (self.window_size+1)//2
        else:
            self.window_shift = window_shift
        self.cpc_samples_per_patent = cpc_samples_per_patent
        self.debug_max_patents = debug_max_patents
        self.n_patents_per_window = n_patents_per_window

    @property
    def year_ranges(self) -> Iterable[list[int]]:
        """Year ranges for the simulation specification."""
        cur_start = self.year_start - ((
            self.year_start-STARTING_YEAR+10000*self.window_shift) % self.window_shift)
        cur_end = cur_start + self.window_size
        while cur_end <= self.year_end:
            yield list(range(cur_start, cur_end))
            cur_start += self.window_shift
            cur_end += self.window_shift

    @property
    def name(self) -> str:
        """Identifier of the simulation specifications.

        This is mainly used to check whether a new run is compatible.
        Different values for `year_start` and `year_end` should be compatible.
        """
        return (f"s{STARTING_YEAR}-w{self.window_size}-"
                f"c{self.cpc_samples_per_patent}-d{self.debug_max_patents}-"
                f"n{self.n_patents_per_window}")

    def check_file(self, output_fp: FileType) -> bool:
        """Check whether the output file has a simulation specification.

        If it doesn't have one, insert the supplied specification.

        Arguments
        ---------
        output_fp:
            File to check.
        sim_spec:
            Specification to check.

        Returns
        -------
            Whether the file now has the same specification.
        """
        with DataModel(output_fp, read_only=False) as data:
            try:
                return data.handle.attrs["sim_spec"] == self.name
            except KeyError:
                data.handle.attrs["sim_spec"] = self.name
        return True
