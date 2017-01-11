import mmap
import numpy as np


class DcimgFile(object):
    """A DCIMG file (Hamamatsu format), memory-mapped.

    After use, call the close() method to release resources properly.
    """

    def __init__(self, file_name=None):
        self.file_name = file_name
        self.mm = None  #: memory-mapped array
        self.nfrms = None  #: number of frames
        self.file_size = None
        self.footer_loc = None
        self.byte_depth = None  #: number of bytes per pixel
        self.x_size = None
        self.y_size = None
        self.bytes_per_row = None
        self.bytes_per_img = None
        self.npa = None  #: memory-mapped numpy array
        self.dtype = None
        self.images_per_layer = 5

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
        return (self.nfrms, self.x_size, self.y_size)

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
        self.file_size = int.from_bytes(self.mm[index:index + 4],
                                        byteorder="little")

        index = 120
        self.footer_loc = int.from_bytes(self.mm[index:index + 4],
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
        self.x_size = int.from_bytes(self.mm[index:index + 4],
                                     byteorder="little")

        index = 168
        self.bytes_per_row = int.from_bytes(self.mm[index:index + 4],
                                            byteorder="little")

        index = 172
        self.y_size = int.from_bytes(self.mm[index:index + 4],
                                     byteorder="little")

        index = 176
        self.bytes_per_img = int.from_bytes(self.mm[index:index + 4],
                                            byteorder="little")

        if self.bytes_per_row != self.byte_depth * self.y_size:
            e_str = "bytes_per_row ({bytes_per_row}) " \
                    "!= byte_depth ({byte_depth}) * nrows ({y_size})" \
                .format(**vars(self))
            raise RuntimeError(e_str)

        if self.bytes_per_img != self.bytes_per_row * self.y_size:
            e_str = "bytes per img ({bytes_per_img}) != nrows ({y_size}) * " \
                    "bytes_per_row ({bytes_per_row})".format(**vars(self))
            raise RuntimeError(e_str)

    def layer(self, index, n_of_images, dtype=None):
        """Return a layer, i.e. a stack of images.

        Parameters
        ----------
        index : layer index
        n_of_images : number of images per layer
        dtype

        Returns
        -------
        A numpy array of the original type or of dtype, if specified.
        """
        offset = 232 + self.bytes_per_img * n_of_images * index
        a = np.ndarray((n_of_images, self.x_size, self.y_size), self.dtype,
                       self.mm, offset)
        if dtype is None:
            return a
        return a.astype(dtype)
