# Based on information gathered from:
# https://github.com/StuartLittlefair/dcimg/blob/master/dcimg/Raw.py
# hamamatsuOrcaTools: https://github.com/orlandi/hamamatsuOrcaTools
# Python Microscopy: http://www.python-microscopy.org
#                    https://bitbucket.org/david_baddeley/python-microscopy

# Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>

"""This module provides the `DCIMGFile` class for accessing Hamamatsu DCIMG
files."""

import logging
import operator
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__version__ = '0.6.0.post1'


class DCIMGFile(object):
    """A DCIMG file (Hamamatsu format), memory-mapped.

    This class provides an interface for reading 3D Hamamatsu DCIMG files.

    Usage is pretty straightforward. First of all, create a `DCIMGFile` object:

    >>> my_file = DCIMGFile('input_file.dcimg')
    >>> my_file
    <DCIMGFile shape=2450x2048x2048 dtype=<class 'numpy.uint16'>\
file_name=input_file.dcimg>

    Image data can then be accessed using NumPy's basic indexing:

    >>> my_file[-10, :5, :5]
    array([[101, 104, 100,  99,  89],
           [103, 102, 103,  99, 102],
           [101, 104,  99, 108,  98],
           [102, 111,  99, 111,  95],
           [103,  98,  99, 104, 106]], dtype=uint16)

    Other convenience methods for accessing image data are: `zslice`,
    `zslice_idx`, `frame` and `whole`.

    `DCIMGFile` supports context managers:

    >>> with DCIMGFile('input_file.dcimg') as f:
    >>>     a = f[800, ...]


    .. seealso:: NumPy's basic indexing: `numpy:arrays.indexing`

    To access the whole file as a :external+dask:doc:`Dask Array<array>`,
    use `as_dask_array()`:

    >>> my_file = DCIMGFile('input_file.dcimg')
    >>> da = my_file.as_dask_array()
    """

    FILE_HDR_DTYPE = [
        ('file_format', 'S8'),
        ('format_version', '<u4'),  # 0x08
        ('skip', '5<u4'),           # 0x0c
        ('nsess', '<u4'),           # 0x20 ?
        ('nfrms', '<u4'),           # 0x24
        ('header_size', '<u4'),     # 0x28 ?
        ('skip2', '<u4'),           # 0x2c
        ('file_size', '<u8'),       # 0x30
        ('skip3', '2<u4'),          # 0x38
        ('file_size2', '<u8'),      # 0x40, repeated
    ]

    SESS_HDR_DTYPE = [
        ('session_size', '<u8'),  # including footer
        ('skip1', '6<u4'),
        ('nfrms', '<u4'),
        ('byte_depth', '<u4'),
        ('skip2', '<u4'),
        ('xsize', '<u4'),
        ('bytes_per_row', '<u4'),
        ('ysize', '<u4'),
        ('bytes_per_img', '<u4'),
        ('skip3', '2<u4'),
        ('offset_to_data', '<u4'),
        ('session_data_size', '<u8'),  # header_size + x*y*byte_depth*nfrms
    ]

    SESSION_FOOTER_DTYPE = [
        ('format_version', '<u4'),
        ('skip0', '<u4'),
        ('offset_to_2nd_struct', '<u8'),
        ('skip1', '2<u4'),
        ('offset_to_offset_to_end_of_data', '<u8'),
        ('skip2', '2<u4'),
        ('footer_size', '<u4'),
        ('skip3', '<u4'),

        # an almost empty part after the footer
        # contains 'offset_to_end_of_data', 0x00000000 0x00000000
        # repeated 2 * nfrms times
        ('2nd_footer_size', '<u4'),  # = 2 * nfrms * 16

        ('skip4', '19<u4'),
        ('offset_to_end_of_data', '<u8'),  # sum of the two offsets above
        ('skip5', '<u8'),  # sum of the two offsets above
        ('offset_to_end_of_data_again', '<u8'),  # repeated
        ('skip6', '<u8'),  # repeated
    ]

    SESSION_FOOTER2_DTYPE = [
        ('offset_to_offset_to_timestamps', '<u8'),
        ('skip0', '2<u4'),
        ('offset_to_offset_to_frame_counts', '<u8'),
        ('skip1', '2<u4'),
        ('offset_to_offset_to_4px', '<u8'),
        ('skip2', '2<u4'),
        ('offset_to_frame_counts', '<u8'),
        ('skip3', '2<u4'),
        ('offset_to_timestamps', '<u8'),
        ('skip4', '4<u4'),
        ('offset_to_4px', '<u8'),
        ('skip5', '<u4'),
        ('4px_offset_in_frame', '<u4'),

        # this is zero if there is no 4px correction info in the footer
        # (maybe because of cropping, so the first line is not included) and
        # is 8 if there is 4px correction info stored in the footer. Might be
        # the size in bytes of the 4px correction for each frame (8 = 4 * 2)
        ('4px_size', '<u8'),
    ]

    # newer versions of the dcimg format have a different header
    NEW_SESSION_HEADER_DTYPE = [
        ('session_size', '<u8'),
        ('skip1', '13<u4'),
        ('nfrms', '<u4'),
        ('byte_depth', '<u4'),
        ('skip2', '<u4'),
        ('xsize', '<u4'),
        ('ysize', '<u4'),
        ('bytes_per_row', '<u4'),
        ('bytes_per_img', '<u4'),
        ('skip3', '2<u4'),
        ('offset_to_data', '<u8'),
        ('skip4', '5<u4'),
        ('frame_footer_size', '<u4'),
    ]

    NEW_FRAME_FOOTER_CAMLINK_DTYPE = [
        ('progressive_number', '<u4'),
        ('timestamp', '<u4'),
        ('timestamp_frac', '<u4'),
        ('4px', '<u8'),
        ('zeros', '3<u4'),
    ]

    NEW_FRAME_FOOTER_USB_DTYPE = [
        ('progressive_number', '<u4'),
        ('timestamp', '<u4'),
        ('timestamp_frac', '<u4'),
        ('zeros', '<u4'),
    ]

    NEW_CROP_INFO = [
        ('x0', '<u2'),
        ('xsize', '<u2'),
        ('y0', '<u2'),
        ('ysize', '<u2'),
    ]

    FMT_OLD = 1
    FMT_NEW = 2

    def __init__(self, file_path=None):
        self.mm = None
        """a `numpy.memmap` object with the raw contents of the DCIMG file."""
        self.mma = None
        """memory-mapped `numpy.ndarray` of the image data, without 4px
        correction."""
        self._deep_copy_enabled = None

        self._file_header = None
        self._sess_header = None
        self._sess_footer = None
        self._sess_footer2 = None
        self._ts_data = None  #: timestamp data
        self._fs_data = None  #: framestamp data
        self.file_path = file_path
        if file_path:
            self.file_path = Path(file_path)
        self.fmt_version = None
        self.x0 = 0
        self.y0 = 0
        self.binning = 1
        self._target_line = -1  #: target line for 4px correction

        self.first_4px_correction_enabled = True
        """Some DCIMG files store, in a separate area, the true values of the
        first 4 pixels of a specific line of each frame (`_target_line`; the
        line is given by the file footer, it is e.g. line 1023 for many
        camera-link files). This switch enables retrieving those 4 pixels. If
        False, those pixels are set to 0. If None, they are left unchanged.
        When the `dtype` is `uint8`, this affects the first 8 pixels (8 bytes
        in any case). Defaults to True."""

        self._4px = None
        """A `numpy.ndarray` of shape (`nfrms`, 4), or (`nfrms`, 8) when the
        `dtype` is `uint8`, containing the corrected pixels of the target
        line."""

        if file_path is not None:
            self.open()

    def __repr__(self):
        return '<DCIMGFile file_path="{}" shape={} dtype={}>'.format(
            self.file_path, self.shape, self.dtype)

    def __del__(self):
        self.close()

    def __enter__(self):
        self.deep_copy_enabled = True
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def deep_copy_enabled(self):
        return self._deep_copy_enabled

    @deep_copy_enabled.setter
    def deep_copy_enabled(self, value):
        self._deep_copy_enabled = value

    @property
    def file_size(self):
        """File size in bytes."""
        return int(self._file_header['file_size'][0])

    @property
    def nfrms(self):
        """Number of frames (Z planes), same as `zsize`."""
        return int(self._sess_header['nfrms'][0])

    @property
    def byte_depth(self):
        """Number of bytes per pixel."""
        return int(self._sess_header['byte_depth'][0])

    @property
    def dtype(self):
        """NumPy numerical dtype."""
        if self.byte_depth == 1:
            return np.dtype(np.uint8)
        elif self.byte_depth == 2:
            return np.dtype(np.uint16)

    @property
    def xsize(self):
        return int(self._sess_header['xsize'][0])

    @property
    def ysize(self):
        return int(self._sess_header['ysize'][0])

    @property
    def zsize(self):
        return self.nfrms

    @property
    def bytes_per_row(self):
        return int(self._sess_header['bytes_per_row'][0])

    @property
    def bytes_per_img(self):
        return int(self._sess_header['bytes_per_img'][0])

    @property
    def shape(self):
        """Shape of the whole image stack.

        Returns
        -------
        tuple
            (`zsize`, `ysize`, `xsize`)
        """
        return self.nfrms, self.ysize, self.xsize

    @property
    def _header_size(self):
        return int(self._file_header['header_size'][0])

    @property
    def _session_footer_offset(self):
        sess_data_size = None
        if self.fmt_version == self.FMT_OLD:
            sess_data_size = int(self._sess_header['session_data_size'][0])
        elif self.fmt_version == self.FMT_NEW:
            sess_data_size = \
                self._sess_header['offset_to_data'][0] \
                + (int(self._sess_header['bytes_per_img'][0] + 8) * self.nfrms)
        return self._header_size + sess_data_size

    def open(self, file_path=None):
        self.close()
        if file_path is not None:
            self.file_path = Path(file_path)

        self.mm = np.memmap(self.file_path, mode='r')

        try:
            self._parse_header()
            self._parse_footer()
        except ValueError:
            self.close()
            raise

        bd = self.byte_depth
        data_strides = None
        data_offset = (int(self._file_header['header_size'][0])
                       + int(self._sess_header['offset_to_data'][0]))
        frame_footer_size = None
        if self.fmt_version == self.FMT_OLD:
            if self._has_4px_data:
                offset = self._session_footer_offset \
                         + int(self._sess_footer2['offset_to_4px'][0])
                self._4px = np.ndarray(
                    (self.nfrms, 4), self.dtype, self.mm, offset)
            data_strides = (self.bytes_per_img, self.bytes_per_row, bd)
        elif self.fmt_version == self.FMT_NEW:
            frame_footer_size = int(self._sess_header['frame_footer_size'][0])
            strides = (self.bytes_per_img + frame_footer_size, bd)
            if self._has_4px_data:
                self._4px = np.ndarray((self.nfrms, 8 // self.byte_depth), self.dtype, self.mm,
                                       data_offset + self.bytes_per_img + 12,
                                       strides)
            padding = self.bytes_per_img - self.xsize * self.ysize * bd
            padding //= self.ysize
            data_strides = (self.bytes_per_img + frame_footer_size,
                            self.xsize * bd + padding,
                            bd)

        self.mma = np.ndarray(self.shape, self.dtype, self.mm,
                              data_offset, data_strides)

        if self.fmt_version == DCIMGFile.FMT_OLD:
            # framestamp offset
            offset = int(self._session_footer_offset + 272)
            self._fs_data = np.ndarray(self.nfrms, np.uint32, self.mm, offset)

            # timestamp offset
            offset += 4 * self.nfrms
            self._ts_data = np.ndarray(
                (self.nfrms, 2), np.uint32, self.mm, offset)
        elif self.fmt_version == DCIMGFile.FMT_NEW:
            # framestamps
            offset = (self._header_size
                      + int(self._sess_header['offset_to_data'][0])
                      + self.bytes_per_img)
            strides = self.bytes_per_img + frame_footer_size
            self._fs_data = np.ndarray(
                self.nfrms, np.uint32, self.mm, offset, strides)

            # timestamps
            offset += 4
            strides = (self.bytes_per_img + frame_footer_size, 4)
            self._ts_data = np.ndarray(
                (self.nfrms, 2), np.uint32, self.mm, offset, strides)

        self.compute_target_line()

    def compute_target_line(self):
        if self.fmt_version == DCIMGFile.FMT_OLD:
            if self._has_4px_data:
                self._target_line = (
                    int(self._sess_footer2['4px_offset_in_frame'][0])
                    // self.bytes_per_row)
            else:
                self._target_line = -1
        elif self._file_header['format_version'] == 0x2000000:
            self._target_line = 0
        else:
            self._target_line = (1023 - self.y0) // self.binning

    def close(self):
        self.mm = None  # Close the memmap.

    def _parse_header(self):
        self._file_header = np.ndarray((1,), self.FILE_HDR_DTYPE, self.mm)

        if not self._file_header['file_format'] == b'DCIMG':
            raise ValueError('Invalid DCIMG file')

        if self._file_header['format_version'] == 0x7:
            sess_dtype = self.SESS_HDR_DTYPE
            self.fmt_version = self.FMT_OLD
        elif self._file_header['format_version'] == 0x1000000 or self._file_header['format_version'] == 0x2000000:
            self.fmt_version = self.FMT_NEW
            sess_dtype = self.NEW_SESSION_HEADER_DTYPE
        else:
            raise ValueError('Invalid DCIMG format version: 0x{:04x}'.format(
                self._file_header['format_version'][0]))

        self._sess_header = np.ndarray((1,), sess_dtype, self.mm,
                                       offset=self._header_size)

        if self.fmt_version == self.FMT_NEW:
            i = self._header_size + 712
            crop_info = np.ndarray((1,), self.NEW_CROP_INFO, self.mm, i)

            self.x0 = int(crop_info['x0'][0])
            self.y0 = int(crop_info['y0'][0])
            binning_x = crop_info['xsize'][0] // self.xsize
            binning_y = crop_info['ysize'][0] // self.ysize

            if binning_x != binning_y:
                raise ValueError('different binning in X and Y')

            if binning_x > 0:
                self.binning = binning_x

        if self.byte_depth != 1 and self.byte_depth != 2:
            raise ValueError(
                "Invalid byte-depth: {}".format(self.byte_depth))

        if self.bytes_per_img != self.bytes_per_row * self.ysize:
            e_str = 'invalid value for bytes_per_img'
            raise ValueError(e_str)

    def _parse_footer(self):
        if self.fmt_version != self.FMT_OLD:
            return

        self._sess_footer = np.ndarray((1,), self.SESSION_FOOTER_DTYPE, self.mm,
                                       self._session_footer_offset)

        offset = (self._session_footer_offset
                  + int(self._sess_footer['offset_to_2nd_struct'][0]))

        self._sess_footer2 = \
            np.ndarray((1,), self.SESSION_FOOTER2_DTYPE, self.mm, offset)

    @property
    def _has_4px_data(self):
        """
        Whether the footer contains 4px correction

        Returns
        -------
        bool
        """
        if self.fmt_version == self.FMT_NEW:
            if self._sess_header['frame_footer_size'] >= 512:
                return True
            return np.dtype(self.NEW_FRAME_FOOTER_CAMLINK_DTYPE).itemsize == self._sess_header['frame_footer_size']

        # maybe this is sufficient
        # return int(self._sess_footer2['4px_size']) > 0

        footer_size = int(self._sess_footer['footer_size'][0])
        offset_to_4px = int(self._sess_footer2['offset_to_4px'][0])

        return footer_size == offset_to_4px + 8 * self.nfrms

    def __getitem__(self, item):
        """Allow to access image data using NumPy's basic indexing."""
        a = self.mma[item]

        if self.deep_copy_enabled:
            a = np.copy(a)

        if self.first_4px_correction_enabled is None:
            return a

        if a.size == 0 or not self._has_4px_data:
            return a

        z, y, x = self._expand_index(item)

        # the 4px correction applies to row `_target_line`, columns 0..n-1
        n = self._4px.shape[-1]
        rows = range(y, y + 1) if isinstance(y, int) else y
        if self._target_line not in rows:
            return a
        if isinstance(x, int):
            if x >= n:
                return a
            cols = x
        else:
            cols = [c for c in x if c < n]
            if not cols:
                return a

        a = np.array(a)  # writeable copy, also when a is a NumPy scalar

        a_index = []
        if not isinstance(z, int):
            z = slice(z.start, z.stop if z.stop >= 0 else None, z.step)
            a_index.append(slice(None))
        if not isinstance(y, int):
            a_index.append(rows.index(self._target_line))
        if not isinstance(x, int):
            a_index.append([i for i, c in enumerate(x) if c < n])

        if self.first_4px_correction_enabled:
            a[tuple(a_index)] = self._4px[z][..., cols]
        else:
            a[tuple(a_index)] = 0

        return a[()] if a.ndim == 0 else a

    def _expand_index(self, item):
        """Expand a basic index into one entry per axis (Z, Y, X).

        Integers are made non-negative and slices become `range` objects
        holding the selected indices, so that each entry tells exactly which
        frames, rows or columns `item` selects.
        """
        if not isinstance(item, tuple):
            item = (item,)

        expanded = []
        for i in item:
            if i is Ellipsis:
                expanded.extend([slice(None)] * (3 - len(item) + 1))
            else:
                expanded.append(i)
        expanded.extend([slice(None)] * (3 - len(expanded)))

        out = []
        for i, size in zip(expanded, self.shape):
            if isinstance(i, slice):
                out.append(range(*i.indices(size)))
                continue
            try:
                i = operator.index(i)
            except TypeError:
                raise TypeError("Invalid type: {}".format(type(i)))
            out.append(i + size if i < 0 else i)
        return out

    @property
    def framestamps(self):
        """Framestamps of all frames.

        Returns
        -------
        `numpy.ndarray`
            A numpy array of dtype `numpy.uint32` with framestamps.
        """
        return self._fs_data

    @property
    def timestamps(self):
        """Timestamps of all frames.

        Returns
        -------
        `numpy.ndarray`
            A numpy array of dtype `numpy.datetime64` with frame timestamps.
        """
        return np.asarray([self.ts(i) for i in range(self.nfrms)])

    def ts(self, frame):
        """
        Timestamp of a single frame.

        Parameters
        ----------
        frame : int
            Frame index

        Returns
        -------
        `numpy.datetime64`
        """
        whole = int.from_bytes(self._ts_data[frame, 0], 'little')
        fraction = int.from_bytes(self._ts_data[frame, 1], 'little')
        return np.datetime64(whole * 10**6 + fraction, 'us')

    @staticmethod
    def _args_to_slice(arg1, arg2=None, step=None):
        myslice = slice(arg1, arg2, step)
        if arg2 is None and step is None:
            myslice = slice(arg1)
        return myslice

    def zslice(self, arg1, arg2=None, step=None, dtype=None, copy=True):
        """Return a slice along `Z`, i.e.\  a substack of frames.

        Parameters
        ----------
        arg1 : int
            Mandatory argument, will be passed to `python:slice`
            If arg2 and step are both None, it will be passed as `slice(arg1)`,
            i.e. it would act as the stop argument.
        arg2 : int
            If not null, will be passed as `slice(arg1, arg2, step)`
        step : int
            If not null, will be passed as `slice(arg1, arg2, step)`
        dtype
        copy : bool
            If True, the requested slice is copied to memory. Otherwise a
            memory mapped array is returned.

        Returns
        -------
        `numpy.ndarray`
            A numpy array of the original type or of `dtype`, if specified. The
            shape of the array is (`end_frame` - `start_frame`, `ysize`,
            `xsize`).
        """
        old_copy = self.deep_copy_enabled
        self.deep_copy_enabled = copy
        myslice = self._args_to_slice(arg1, arg2, step)
        a = self[myslice.start:myslice.stop:myslice.step]
        if dtype is not None:
            a = a.astype(dtype)
        self.deep_copy_enabled = old_copy

        return a

    def zslice_idx(self, index, frames_per_slice=1, dtype=None, copy=True):
        """Return a slice, i.e.\  a substack of frames, by index.

        Parameters
        ----------
        index : int
            slice index
        frames_per_slice : int
            number of frames per slice
        dtype
        copy : see `zslice`

        Returns
        -------
        `numpy.ndarray`
            A numpy array of the original type or of `dtype`, if specified. The
            shape of the array is  (`frames_per_slice`, `ysize`, `xsize`).
        """
        start_frame = index * frames_per_slice
        end_frame = start_frame + frames_per_slice
        return self.zslice(start_frame, end_frame, 1, dtype, copy)

    def whole(self, dtype=None, copy=True):
        """Convenience function to retrieve the whole stack.

        Equivalent to call `zslice_idx` with `index` = 0 and
        `frames_per_slice` = `nfrms`

        Parameters
        ----------
        dtype
        copy : see `zslice`

        Returns
        -------
        `numpy.ndarray`
            A numpy array of the original type or of dtype, if specified. The
            shape of the array is `shape`.
        """
        return self.zslice_idx(0, self.nfrms, dtype, copy)

    def frame(self, index, dtype=None, copy=True):
        """Convenience function to retrieve a single frame (Z plane).

        Same as calling `zslice_idx` and squeezing.

        Parameters
        ----------
        index : int
            frame index
        dtype
        copy : see `zslice`

        Returns
        -------
        `numpy.ndarray`
            A numpy array of the original type or of `dtype`, if specified. The
            shape of the array is (`ysize`, `xsize`).
        """
        return np.squeeze(self.zslice_idx(index, dtype=dtype, copy=copy))

    def as_dask_array(self):
        """Return the whole file as a :external+dask:doc:`Dask Array<array>`.

        :external+dask:doc:`Dask <index>` needs to be installed.
        """
        import dask.array as da
        from dask import delayed

        delayed_reader = delayed(self.frame)
        lazy_arrays = [delayed_reader(f) for f in range(self.nfrms)]
        dask_arrays = [da.from_delayed(delayed_reader, shape=(self.ysize, self.xsize), dtype=self.dtype)
                       for delayed_reader in lazy_arrays]
        return da.stack(dask_arrays, axis=0)
