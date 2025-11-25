# Transcripts

High-performance Python/C extension for extracting ordinal patterns, transcripts, delay vectors, and related information-theoretic measures from time series. The module exposes a set of optimized functions implemented in C and wrapped for Python.

## Features

* Delay vector construction
* Ordinal pattern generation
* Transcript calculation between two time series
* Permutation entropy
* Order class analysis
* Coupling complexity
* Transcript mutual information
* Jensen–Shannon divergence for symbolic series

## Installation

```bash
python setup.py install
```

## Available Functions

### `delay_vectors(data, delay, dim)`

Generate delay vectors from a 1D time series.

**Args**

* `data`: 1D array-like
* `delay`: embedding delay
* `dim`: embedding dimension

**Returns**

* 2D NumPy array of delay vectors

### `get_transcripts(data1, data2, delay, dim)`

Generate transcripts [1] between two time series.

**Args**

* `data1`: source series
* `dat2`: target series
* `deöay`: embedding delay
* `dim`: embedding dimension

**Returns**

* int32 NumPy array of transcripts

### `permutation_entropy(data, delay, dim)`

Calculate permutation entropy [2] of.

**Args**

* `data`: 1D series
* `delay`: embedding delay
* `dim`: embedding dimension

**Returns**

* int32 symbol array


### `find_symbols(data, tau, p)`

Compute ordinal patterns [2] from a time series.

**Args** same as above.

**Returns** int32 symbol array.

### `find_order_classes(data, tau, p)`
 
 Create ordinal patterns and calculate resp. order classes [1] from time series.

**Args** same as above.

**Returns** int32 symbol array.

### `transcript_mi(data1, data2, symbol_delay, delay, dim)`

Compute asymmetric mutual information based on transcripts [3].

**Args**

*`data1`: source time series
*`data2`: target time series
*`symbol_delay`: time delay step for asymmetric MI
*`delay`: embedding delay
*`dim`: embedding dimension

**Returns** int32 symbol array.

### `coupling_complexity(data, delay, dim)`

Compute coupling complexity [4] between multiple time series.

* `data`: 2D NumPy array of shape (len_series, n_series) created by np.column_stack((data1, data2, ...))
* `delay`: embedding delay
* `dim`: embedding dimension

**Returns** int32 symbol array.

### `coupling_complexity_symbols(symbols)`

Compute coupling complexity directly from symbol series.

**Args**

*`symbols`: 3D NumPy array of shape (len_series, dim, n_series) created by np.column_stack((symbols1, symbols2, ...))

**Returns** int32 symbol array.

### `ocs_from_symbol_series(symbols)`

Compute order classes from an ordinal pattern series.

**Args**

* `symbols`: 2D ordinal pattern array

**Returns** int64 array.

### `find_order_class(data)`

Calculate order class for a single ordinal pattern.

**Args**

* `pattern`: ordinal pattern

**Returns** int32 array.

### `JS_C(ts1, ts2, tau, p)`

Compute the Jensen–Shannon divergence for order classes between two time series.

**Args**

* `ts1`, `ts2`: 2D time series arrays
* `tau`, `p`: delay and dimension

**Returns**

* float JS-divergence value

## Example

```python
import numpy as np
import transcripts as tr

data1 = np.random.randn(1000)
data2 = np.random.randn(1000)
data3 = np.random.randn(1000)
vecs = tr.delay_vectors(data1, tau=1, p=3)
syms = tr.find_symbols(data1, tau=1, p=3)
trs = tr.get_transcripts(data1,data2,tau=1,p=3)
cc = sb.coupling_complexity(np.column_stack((data1,data2,data3)),tau=1,p=3)
```

## Module Definition

The C extension defines the method table and module structure using `PyModuleDef` and `PyMethodDef` for efficient binding.

## References

- [1] Monetti, R., Bunk, W. & Aschenbrenner, T. Characterizing synchronization in time series using information measures extracted from symbolic representations. Phys. Rev. E 79, 046207 (2009)
- [2] Bandt, C. & Pompe, B. “Permutation Entropy: A Natural Complexity Measure for Time Series.” Phys. Rev. Lett. 88, 174102 (2002).
- [3] Monetti, R., Bunk, W., Aschenbrenner, T., Springer, S. & Amigó, J. M. Information directionality in coupled time series using transcripts. Phys. Rev. E 88, 022911 (2013).
- [4] Monetti, R., Amigó, J. M., Aschenbrenner, T. & Bunk, W. Permutation complexity of interacting dynamical systems. Eur. Phys. J. Spec. Top. 222, 421–436 (2013)

## License

MIT
