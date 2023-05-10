# Simulation specification

The `SimulationSpecification` class is central to doing embedding/novelty runs
in the docembedder package. It determines which patents will be used for training
and fitting the patents. The main idea is to seperate the processing into multiple
windows, such that patents in similar years are trained and fitted together. These
windows will shift by a specified amount. For example, with a window shift of 1 and a
window size of 5, we could have the following windows: 1900-1904, 1901-1905, etc.

An instance of the specification as follows:

```python
sim_spec = SimulationSpecification(
	year_start=1900,  # Starting year of the processing
	year_end=1950,  # End year of the processing (last year included will be 1949)
	window_size=1,  # Size of the window (number of years included)
	window_shift=1,  # Shift between subsequent windows.
)
```
