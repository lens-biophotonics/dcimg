# Based on:
# https://github.com/StuartLittlefair/dcimg/blob/master/dcimg/Raw.py
# hamamatsuOrcaTools: https://github.com/orlandi/hamamatsuOrcaTools
# Python Microscopy: http://www.python-microscopy.org
#                    https://bitbucket.org/david_baddeley/python-microscopy

# Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>

import mmap
from math import pow, log10, floor

import semver
import numpy as np

major = 0
minor = 3
patch = 0

prerelease = None
build = None

version = semver.format_version(major, minor, patch)
full_version = semver.format_version(major, minor, patch, prerelease, build)


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
        self.mm = None  #: memory-mapped array
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
                val += fraction * pow(10, -(floor(log10(fraction)) + 1))
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

    def slice(self, start_frame, end_frame=None, dtype=None, copy=True):
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

        if end_frame is None:
            end_frame = start_frame + 1

        nframes = end_frame - start_frame

        offset = 232 + self.bytes_per_img * start_frame
        a = np.ndarray(
            (nframes, self.ysize, self.xsize), self.dtype, self.mm, offset)

        if copy:
            a = np.copy(a)

        # retrieve the first 4 pixels of each frame, which are stored in the
        # file footer. Will overwrite [0000, FFFF, 0000, FFFF, 0000] at the
        # beginning of the frame.
        if self.retrieve_first_4_pixels:
            index = (self.session_footer_offset + 272
                     + self.nfrms * (4 + 8)  # 4: frame count, 8: timestamp
                     + 4 * self.byte_depth * start_frame)
            for i in range(0, nframes):
                px = np.ndarray((1, 1, 4), self.dtype, self.mm, index)
                a[i, 0, 0:4] = px
                index += 4 * self.byte_depth
        else:
            a[:, 0, 0:4] = 0

        if dtype is None:
            return a
        return a.astype(dtype)

    def slice_idx(self, index, frames_per_slice=1, dtype=None, copy=True):
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
        return self.slice(start_frame, end_frame, dtype, copy)

    def whole(self, dtype=None, copy=True):
        """Convenience function to retrieve the whole stack.

        Equivalent to call :func:`slice_idx` with `index` = 0 and
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
        return self.slice_idx(0, self.nfrms, dtype, copy)

    def frame(self, index, dtype=None, copy=True):
        """Convenience function to retrieve a single frame (Z plane).

        Same as calling :func:`slice_idx` and squeezing.

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
        return np.squeeze(self.slice_idx(index, dtype=dtype, copy=copy))
