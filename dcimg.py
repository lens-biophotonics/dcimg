import mmap
import numpy as np
from math import pow, log10, floor


class DCIMGFile(object):
    """A DCIMG file (Hamamatsu format), memory-mapped.

    After use, call the close() method to release resources properly.
    """

    def __init__(self, file_name=None):
        self.file_name = file_name
        self.mm = None  #: memory-mapped array
        self.file_size = None
        self.footer_offset = None
        self.byte_depth = None  #: number of bytes per pixel
        self.xsize = None
        self.ysize = None
        self.nfrms = None  #: number of frames
        self.bytes_per_row = None
        self.bytes_per_img = None
        self.npa = None  #: memory-mapped numpy array
        self.dtype = None

    def open(self, file_name=None):
        if file_name is None:
            file_name = self.file_name

        with open(file_name, 'r') as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            if mm[:5] != b"DCIMG":
                raise RuntimeError("Invalid DCIMG file")

        self.mm = mm
        self._parse_header()

        self.npa = np.ndarray(self.shape, self.dtype, self.mm, 232)

    @property
    def shape(self):
        return (self.nfrms, self.xsize, self.ysize)

    @property
    def timestamp_offset(self):
        return self.footer_offset + 272 + 4 * self.nfrms

    def close(self):
        if self.mm is not None:
            self.mm.close()
        self.mm = None
        self.npa = None

    def _parse_header(self):
        bytes_to_skip = 4 * int.from_bytes(self.mm[8:12], byteorder="little")

        index = 8 + bytes_to_skip
        self.nfrms = int.from_bytes(self.mm[index:index + 4],
                                    byteorder="little")

        index = 48
        self.file_size = int.from_bytes(self.mm[index:index + 8],
                                        byteorder="little")

        index = 156
        self.byte_depth = int.from_bytes(self.mm[index:index + 4],
                                         byteorder="little")

        if self.byte_depth == 1:
            self.dtype = np.uint8
        elif self.byte_depth == 2:
            self.dtype = np.uint16
        else:
            raise RuntimeError(
                "Invalid byte-depth: {}".format(self.byte_depth))

        index = 164
        self.xsize = int.from_bytes(self.mm[index:index + 4],
                                    byteorder="little")

        index = 168
        self.bytes_per_row = int.from_bytes(self.mm[index:index + 4],
                                            byteorder="little")

        index = 172
        self.ysize = int.from_bytes(self.mm[index:index + 4],
                                    byteorder="little")

        index = 176
        self.bytes_per_img = int.from_bytes(self.mm[index:index + 4],
                                            byteorder="little")

        index = 192
        self.footer_offset = int.from_bytes(self.mm[index:index + 8],
                                            byteorder="little")
        index = 40
        self.footer_offset += int.from_bytes(self.mm[index:index + 8],
                                             byteorder="little")

        if self.bytes_per_row != self.byte_depth * self.ysize:
            e_str = "bytes_per_row ({bytes_per_row}) " \
                    "!= byte_depth ({byte_depth}) * nrows ({y_size})" \
                .format(**vars(self))
            raise RuntimeError(e_str)

        if self.bytes_per_img != self.bytes_per_row * self.ysize:
            e_str = "bytes per img ({bytes_per_img}) != nrows ({y_size}) * " \
                    "bytes_per_row ({bytes_per_row})".format(**vars(self))
            raise RuntimeError(e_str)

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

    def layer(self, index, frames_per_layer=1, dtype=None):
        """Return a layer, i.e. a stack of frames.

        Parameters
        ----------
        index : layer index
        frames_per_layer : number of frames per layer
        dtype

        Returns
        -------
        A numpy array of the original type or of dtype, if specified. The
        shape of the array is (nframes, ysize, xsize).
        """
        offset = 232 + self.bytes_per_img * frames_per_layer * index
        a = np.ndarray((frames_per_layer, self.ysize, self.xsize),
                       self.dtype, self.mm, offset)
        if dtype is None:
            return a
        return a.astype(dtype)

    def frame(self, index, dtype=None):
        """Convenience function to retrieve a single layer.

        Same as calling layer() with frames_per_layer=1.

        Parameters
        ----------
        index : layer index
        dtype

        Returns
        -------
        A numpy array of the original type or of dtype, if specified. The
        shape of the array is (ysize, xsize).
        """
        return np.squeeze(self.layer(index), dtype)
