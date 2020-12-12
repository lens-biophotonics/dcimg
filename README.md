# dcimg

This module provides the `DCIMGFile` class for accessing Hamamatsu DCIMG
files.

## Installation
```bash
pip install dcimg
```

## Documentation
The `DCIMGFile` class provides an interface for reading 3D Hamamatsu DCIMG
files.

Usage is pretty straightforward. First of all, create a `DCIMGFile` object:

```python
>>> my_file = DCIMGFile('input_file.dcimg')
>>> my_file
    <DCIMGFile shape=2450x2048x2048 dtype=<class 'numpy.uint16'> file_name=input_file.dcimg>
```
Image data can then be accessed using NumPy's basic indexing:

```python
>>> my_file[-10, :5, :5]
array([[101, 104, 100,  99,  89],
       [103, 102, 103,  99, 102],
       [101, 104,  99, 108,  98],
       [102, 111,  99, 111,  95],
       [103,  98,  99, 104, 106]], dtype=uint16)
```

Other convenience methods for accessing image data are: `zslice`, `zslice_idx`,
`frame` and `whole`.

`DCIMGFile` supports context managers:
```python
with DCIMGFile('input_file.dcimg') as f:
     a = f[800, ...]
```

For further details refer to:
https://lens-biophotonics.github.io/dcimg/
