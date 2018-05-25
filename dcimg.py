# Based on information gathered from:
# https://github.com/StuartLittlefair/dcimg/blob/master/dcimg/Raw.py
# hamamatsuOrcaTools: https://github.com/orlandi/hamamatsuOrcaTools
# Python Microscopy: http://www.python-microscopy.org
#                    https://bitbucket.org/david_baddeley/python-microscopy

# Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>

"""This module provides the `DCIMGFile` class for accessing Hamamatsu DCIMG
files."""

import math
import mmap

import numpy as np

__version__ = '0.5.0'


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
        ('offset_to_data', '1<u4'),
        ('session_data_size', '<u8'),  # header_size + x*y*byte_depth*nfrms
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
    ]

    NEW_FRAME_FOOTER_DTYPE = [
        ('progressive_number', '<u4'),
        ('timestamp', '<u4'),
        ('timestamp_frac', '<u4'),
        ('4px', '<u8'),
        ('zeros', '<u12'),
    ]

    FMT_OLD = 1
    FMT_NEW = 2

    def __init__(self, file_name=None):
        self.mm = None
        """a `mmap.mmap` object with the raw contents of the DCIMG file."""
        self.mma = None
        """memory-mapped `numpy.ndarray` of the image data, without 4px
        correction."""
        self._deep_copy_enabled = None

        self.fileno = None  #: file descriptor
        self.file = None
        self._file_header = None
        self._sess_header = None
        self.file_name = file_name
        self.fmt_version = None

        self.first_4px_correction_enabled = True
        """For some reason, the first 4 pixels of each frame (or the first 4
        pixels of line number 1023 of each frame) are stored in a different
        area in the file. This switch enables retrieving those 4 pixels. If
        False, those pixels are set to 0. If None, they are left unchanged.
        Defaults to True."""

        self._4px = None
        """A `numpy.ndarray` of shape (`nfrms`, 4) containing the first 4
        pixels of each frame."""

        if file_name is not None:
            self.open()

    def __repr__(self):
        return '<DCIMGFile shape={}x{}x{} dtype={} file_name={}>'.format(
            *self.shape, self.dtype, self.file_name)

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
        return self._file_header['file_size'][0]

    @property
    def nfrms(self):
        """Number of frames (Z planes), same as `zsize`."""
        return self._sess_header['nfrms'][0]

    @property
    def byte_depth(self):
        """Number of bytes per pixel."""
        return self._sess_header['byte_depth'][0]

    @property
    def dtype(self):
        """NumPy numerical dtype."""
        if self.byte_depth == 1:
            return np.uint8
        elif self.byte_depth == 2:
            return np.uint16

    @property
    def xsize(self):
        return self._sess_header['xsize'][0]

    @property
    def ysize(self):
        return self._sess_header['ysize'][0]

    @property
    def zsize(self):
        return self.nfrms

    @property
    def bytes_per_row(self):
        return self._sess_header['bytes_per_row'][0]

    @property
    def bytes_per_img(self):
        return self._sess_header['bytes_per_img'][0]

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
        return self._file_header['header_size'][0]

    @property
    def _session_footer_offset(self):
        try:
            sess_data_size = int(self._sess_header['session_data_size'][0])
        except ValueError:
            sess_data_size = self._sess_header['offset_to_data'][0] + (int(
                self._sess_header['bytes_per_img'][0] + 8)
                              * self.nfrms)
        return self._header_size + sess_data_size

    def timestamps(self):
        """Get frame timestamps.

        Returns
        -------
        `numpy.ndarray`
            A numpy array of dtype `numpy.float64` with frame timestamps in
            Unix time plus a fractional part.
        """
        ts = np.zeros(self.nfrms)
        if self.fmt_version == DCIMGFile.FMT_OLD:
            # timestamp offset
            offset = int(self._session_footer_offset + 272 + 4 * self.nfrms)
            data = np.ndarray((self.nfrms, 2), np.uint32, self.mm, offset)
        elif self.fmt_version == DCIMGFile.FMT_NEW:
            offset = int(self._file_header['header_size']
                         + self._sess_header['offset_to_data'][0]
                         + self.bytes_per_img + 4)
            strides = (self.bytes_per_img + 32, 4)
            data = np.ndarray((self.nfrms, 2), np.uint32, self.mm,
                              offset, strides)

        for i in range(0, self.nfrms):
            whole = int.from_bytes(data[i, 0], 'little')

            fraction = int.from_bytes(data[i, 1], 'little')

            val = whole
            if fraction != 0:
                val += fraction * math.pow(
                    10, -(math.floor(math.log10(fraction)) + 1))
            ts[i] = val

        return ts

    def open(self, file_name=None):
        self.close()
        if file_name is None:
            file_name = self.file_name

        f = open(file_name, 'r')
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)

        self.fileno = f.fileno()
        self.file = f
        self.mm = mm

        try:
            self._parse_header()
        except ValueError:
            self.close()
            raise

        if self.fmt_version == self.FMT_OLD:
            offset = (self._session_footer_offset + 272
                      + self.nfrms * (4 + 8))  # 4: frame count, 8: timestamp
            self._4px = np.ndarray((self.nfrms, 4), self.dtype, self.mm, offset)

        offset = int(self._file_header['header_size']
                     + self._sess_header['offset_to_data'][0])
        strides = None
        if self.fmt_version == self.FMT_NEW:
            bd = self.byte_depth
            strides = (self.bytes_per_img + 32, bd)
            self._4px = np.ndarray((self.nfrms, 4), self.dtype, self.mm,
                                   offset + self.bytes_per_img + 12,
                                   strides)

            strides = np.array([
                self.ysize * self.xsize,
                self.xsize,
                1
            ]) * bd
            strides[0] += 32
        self.mma = np.ndarray(self.shape, self.dtype, self.mm, offset, strides)

    def close(self):
        if self.mm is not None:
            self.mm.close()
        del self.mm
        self.mm = None
        if self.file is not None:
            self.file.close()

    def _parse_header(self):
        data = self.mm[0:np.dtype(self.FILE_HDR_DTYPE).itemsize]
        self._file_header = np.fromstring(data, dtype=self.FILE_HDR_DTYPE)

        if not self._file_header['file_format'] == b'DCIMG':
            raise ValueError('Invalid DCIMG file')

        if self._file_header['format_version'] == 0x7:
            sess_dtype = self.SESS_HDR_DTYPE
            self.fmt_version = self.FMT_OLD
        elif self._file_header['format_version'] == 0x1000000:
            self.fmt_version = self.FMT_NEW
            sess_dtype = self.NEW_SESSION_HEADER_DTYPE
        else:
            raise ValueError('Invalid DCIMG format version: 0x{:04x}'.format(
                self._file_header['format_version'][0]))

        self._sess_header = np.zeros(1, dtype=sess_dtype)
        index_from = self._header_size
        index_to = index_from + self._sess_header.nbytes
        self._sess_header = np.fromstring(self.mm[index_from:index_to],
                                          dtype=sess_dtype)

        if self.byte_depth != 1 and self.byte_depth != 2:
            raise ValueError(
                "Invalid byte-depth: {}".format(self.byte_depth))

        if self.bytes_per_row != self.byte_depth * self.ysize:
            e_str = "bytes_per_row ({bytes_per_row}) " \
                    "!= byte_depth ({byte_depth}) * nrows ({y_size})" \
                .format(**vars(self))
            raise ValueError(e_str)

        if self.bytes_per_img != self.bytes_per_row * self.ysize:
            e_str = "bytes per img ({bytes_per_img}) != nrows ({y_size}) * " \
                    "bytes_per_row ({bytes_per_row})".format(**vars(self))
            raise ValueError(e_str)

    def __getitem__(self, item, copy=None):
        """Allow to access image data using NumPy's basic indexing."""
        a = self.mma[item]

        if copy is None:
            deepcopy = self.deep_copy_enabled
        else:
            deepcopy = copy

        if deepcopy:
            a = np.copy(a)

        if self.first_4px_correction_enabled is None:
            return a

        if a.size == 0:
            return a

        # ensure item is a tuple
        if isinstance(item, list):
            item = tuple(item)
        else:
            item = np.index_exp[item]

        # ensure all items are slice objects
        myitem = []
        for i in item:
            if isinstance(i, int):
                start = i
                stop = i + 1
                step = 1
            elif i is Ellipsis:
                for _ in range(0, 3 - len(item) + 1):
                    myitem.append(slice(0, self.shape[len(myitem)], 1))
                continue
            elif isinstance(i, slice):
                start = i.start
                stop = i.stop
                step = i.step if i.step is not None else 1
            else:
                raise TypeError("Invalid type: {}".format(type(i)))

            curr_max = self.shape[len(myitem)]
            if start is None:
                start = 0 if step > 0 else curr_max
            elif start < 0:
                start += curr_max
            elif start > curr_max:
                start = curr_max

            if stop is None:
                stop = curr_max if step > 0 else 0
            elif stop < 0:
                stop += curr_max
            elif stop > curr_max:
                stop = curr_max

            myitem.append(slice(start, stop, step))

        for _ in range(0, 3 - len(myitem)):
            myitem.append(slice(0, self.shape[len(myitem)], 1))

        startx = myitem[2].start
        stopx = myitem[2].stop
        stepx = myitem[2].step

        starty = myitem[1].start
        stopy = myitem[1].stop
        stepy = myitem[1].step

        if self.fmt_version == self.FMT_OLD:
            condition_y = starty == 0 or stopy == 0
        elif self.fmt_version == self.FMT_NEW:
            if stepy > 0:
                condition_y = starty <= 1023 and stopy >= 1023
            elif stepy < 0:
                condition_y = stopy <= 1023 and starty >= 1023

        if condition_y and ((startx >= 0 and startx < 4) or stopx < 4):
            if isinstance(a, self.dtype):
                if self.first_4px_correction_enabled:
                    a = self._4px[myitem[0].start, startx]
                else:
                    a = 0
                return a

            if startx < stopx:
                newstartx = 0
                if stopx > 4:
                    newstopx = int(math.ceil((4 - startx) / abs(stepx)))
                else:
                    newstopx = (stopx - startx) // abs(stepx)
            else:
                newstopx = a.shape[-1]
                if a.shape[-1] < 4:
                    newstartx = 0
                else:
                    newstartx = (a.shape[-1] - 4 // abs(stepx))

            if newstartx == newstopx:
                return np.empty([0])

            newshape = [math.ceil(
                (myitem[i].stop - myitem[i].start) / myitem[i].step)
                for i in range(0, 3)]

            old_shape = a.shape

            a.shape = newshape

            if self.fmt_version == DCIMGFile.FMT_OLD:
                target_line = 0
            else:
                target_line = 1023

            newy = int(math.floor((target_line - starty) / stepy))
            if stepy < 0:
                newy -= 1

            a_index_exp = np.index_exp[..., newy, newstartx:newstopx]

            if self.first_4px_correction_enabled:
                _range = sorted((startx, stopx))
                _4start = max(0, _range[0])
                _4stop = min(4, _range[1])
                _4px = self._4px[item[0], _4start:_4stop:abs(stepx)]

                if stepx < 0:
                    _4px = _4px[..., ::-1]
                a[a_index_exp] = _4px
            else:
                a[a_index_exp] = 0

            a.shape = old_shape

        return a

    def zslice(self, start_frame, end_frame=None, dtype=None, copy=True):
        """Return a slice along `Z`, i.e.\  a substack of frames.

        Parameters
        ----------
        start_frame : int
            first frame to select
        end_frame : int
            last frame to select (noninclusive). If None, defaults to
            `start_frame + 1`
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
        a = self.__getitem__(slice(start_frame, end_frame), copy=copy)
        if dtype is not None:
            a = a.astype(dtype)

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
        return self.zslice(start_frame, end_frame, dtype, copy)

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
