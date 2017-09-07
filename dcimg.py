# Based on:
# https://github.com/StuartLittlefair/dcimg/blob/master/dcimg/Raw.py
# hamamatsuOrcaTools: https://github.com/orlandi/hamamatsuOrcaTools
# Python Microscopy: http://www.python-microscopy.org
#                    https://bitbucket.org/david_baddeley/python-microscopy

# Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>

import math
import mmap

import numpy as np

_major = 0
_minor = 3
_micro = 0

__version__ = '{major}.{minor}.{micro}'.format(major=_major, minor=_minor,
                                               micro=_micro)


class DCIMGFile(object):
    """A DCIMG file (Hamamatsu format), memory-mapped.

    After use, call the close() method to release resources properly.
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
        ('header_size', '1<u4'),
        ('session_data_size', '<u8'),  # header_size + x*y*byte_depth*nfrms
    ]

    def __init__(self, file_name=None):
        self.mm = None
        """a :class:`python:mmap.mmap` object"""
        self.mma = None
        """memory-mapped :class:`numpy.ndarray`"""
        self.deep_copy_enabled = None

        self.fileno = None  #: file descriptor
        self.file = None
        self.file_header = None
        self.sess_header = None
        self.dtype = None
        self.file_name = file_name

        self.retrieve_first_4_pixels = True
        """For some reason, the first 4 pixels of each frame are stored in a
        different area in the file. This switch enables retrieving those 4
        pixels. If False, those pixels are set to 0. Defaults to True."""

        self._4px = None
        """A :class:`numpy.ndarray` of shape (:attr:`nfrms`, 4) containing
        the first 4 pixels of each frame."""

        if file_name is not None:
            self.open()

    def __repr__(self):
        return '<DCIMGFile shape={}x{}x{} dtype={} file_name={}>'.format(
            *self.shape, self.dtype, self.file_name)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def file_size(self):
        return self.file_header['file_size'][0]

    @property
    def nfrms(self):
        return self.sess_header['nfrms'][0]

    @property
    def byte_depth(self):
        """Number of bytes per pixel."""
        return self.sess_header['byte_depth'][0]

    @property
    def xsize(self):
        return self.sess_header['xsize'][0]

    @property
    def ysize(self):
        return self.sess_header['ysize'][0]

    @property
    def bytes_per_row(self):
        return self.sess_header['bytes_per_row'][0]

    @property
    def bytes_per_img(self):
        return self.sess_header['bytes_per_img'][0]

    @property
    def shape(self):
        """Shape of the whole image stack.

        Returns
        -------
        tuple
            (:attr:`nfrms`, :attr:`ysize`, :attr:`xsize`)
        """
        return (self.nfrms, self.ysize, self.xsize)

    @property
    def header_size(self):
        return self.file_header['header_size'][0]

    @property
    def session_footer_offset(self):
        return int(self.header_size + self.sess_header['session_data_size'][0])

    @property
    def timestamp_offset(self):
        return int(self.session_footer_offset + 272 + 4 * self.nfrms)

    @property
    def timestamps(self):
        """A numpy array with frame timestamps."""
        ts = np.zeros(self.nfrms)
        index = self.timestamp_offset
        for i in range(0, self.nfrms):
            whole = int.from_bytes(self.mm[index:index + 4], 'little')
            index += 4

            fraction = int.from_bytes(self.mm[index:index + 4], 'little')
            index += 4

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

        offset = (self.session_footer_offset + 272
                  + self.nfrms * (4 + 8))  # 4: frame count, 8: timestamp
        self._4px = np.ndarray((self.nfrms, 4), self.dtype, self.mm, offset)

        offset = 232
        self.mma = np.ndarray(self.shape, self.dtype, self.mm, offset)

    def close(self):
        if self.mm is not None:
            self.mm.close()
        del self.mm
        self.mm = None
        if self.file is not None:
            self.file.close()

    def _parse_header(self):
        data = self.mm[0:np.dtype(self.FILE_HDR_DTYPE).itemsize]
        self.file_header = np.fromstring(data, dtype=self.FILE_HDR_DTYPE)

        if not self.file_header['file_format'] == b'DCIMG':
            raise ValueError('Invalid DCIMG file')

        self.sess_header = np.zeros(1, dtype=self.SESS_HDR_DTYPE)
        index_from = self.header_size
        index_to = index_from + self.sess_header.nbytes
        self.sess_header = np.fromstring(self.mm[index_from:index_to],
                                         dtype=self.SESS_HDR_DTYPE)

        if self.byte_depth == 1:
            self.dtype = np.uint8
        elif self.byte_depth == 2:
            self.dtype = np.uint16
        else:
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

    def __getitem__(self, item, copy=False):
        a = self.mma[item]

        # ensure item is a tuple
        item = np.index_exp[item]

        # ensure all items are slice objects
        myitem = []
        for i in item:
            if isinstance(i, int):
                myitem.append(slice(i, i + 1))
            elif i is Ellipsis:
                for _ in range(0, 3 - len(item) + 1):
                    myitem.append(slice(0, self.shape[len(myitem)], 1))
            elif isinstance(i, slice):
                myitem.append(i)
            else:
                raise TypeError("Invalid type: {}".format(type(i)))

        for _ in range(0, 3 - len(myitem)):
            myitem.append(slice(0, self.shape[len(myitem)], 1))

        if self.deep_copy_enabled is None:
            deepcopy = copy
        else:
            deepcopy = self.deep_copy_enabled

        if deepcopy:
            a = np.copy(a)

        startx = myitem[-1].start
        if startx is None:
            startx = 0
        stopx = myitem[-1].stop
        if stopx is None:
            stopx = self.shape[-1]
        starty = myitem[-2].start
        if (starty is None or starty == 0) and startx < 4:
            stopx = 4 if stopx > 4 else stopx
            stepx = myitem[-1].step
            if isinstance(a, self.dtype):
                if self.retrieve_first_4_pixels:
                    a = self._4px[myitem[0].start, startx]
                else:
                    a = 0
                return a
            elif len(a.shape) > 1:
                a_index_exp = np.index_exp[
                              ..., 0, :math.ceil((stopx - startx) / stepx)]
            else:
                a_index_exp = np.index_exp[
                              ..., :math.ceil((stopx - startx) / stepx)]
            if self.retrieve_first_4_pixels:
                a[a_index_exp] = self._4px[myitem[0], startx:stopx:stepx]
            else:
                a[a_index_exp] = 0

        return a

    def zslice(self, start_frame, end_frame=None, dtype=None, copy=True):
        """Return a slice along `Z`, i.e.\  a substack of frames.

        Parameters
        ----------
        start_frame : int
            first frame to select
        end_frame : int
            last frame to select (noninclusive). If None, defaults to
            :code:`start_frame + 1`
        dtype
        copy : bool
            If True, the requested slice is copied to memory. Otherwise a
            memory mapped array is returned.

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array of the original type or of `dtype`, if specified. The
            shape of the array is (`end_frame` - `start_frame`, :attr:`ysize`,
            :attr:`xsize`).
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
        copy : see :func:`slice`

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array of the original type or of `dtype`, if specified. The
            shape of the array is  (`frames_per_slice`, :attr:`ysize`,
            :attr:`xsize`).
        """
        start_frame = index * frames_per_slice
        end_frame = start_frame + frames_per_slice
        return self.zslice(start_frame, end_frame, dtype, copy)

    def whole(self, dtype=None, copy=True):
        """Convenience function to retrieve the whole stack.

        Equivalent to call :func:`zslice_idx` with `index` = 0 and
        `frames_per_slice` = :attr:`nfrms`

        Parameters
        ----------
        dtype
        copy : see :func:`slice`

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array of the original type or of dtype, if specified. The
            shape of the array is :attr:`shape`.
        """
        return self.zslice_idx(0, self.nfrms, dtype, copy)

    def frame(self, index, dtype=None, copy=True):
        """Convenience function to retrieve a single frame (Z plane).

        Same as calling :func:`zslice_idx` and squeezing.

        Parameters
        ----------
        index : int
            frame index
        dtype
        copy : see :func:`slice`

        Returns
        -------
        :class:`numpy.ndarray`
            A numpy array of the original type or of `dtype`, if specified. The
            shape of the array is (:attr:`ysize`, :attr:`xsize`).
        """
        return np.squeeze(self.zslice_idx(index, dtype=dtype, copy=copy))
