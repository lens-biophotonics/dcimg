import unittest
from ddt import ddt, data

import numpy as np

from dcimg import DCIMGFile


test_vectors = [
    np.index_exp[...],
    np.index_exp[:, :, :],
    np.index_exp[..., ::-1],
    np.index_exp[..., ::-1, :],
    np.index_exp[::-1, ...],
    np.index_exp[::-1, ::-1, ::-1],
    np.index_exp[..., -10:-21:-1],
    np.index_exp[2, 0, -10:-21:-1],
    np.index_exp[2, 0:4, -10:-21:-1],
    np.index_exp[-4:-7:-1, 0, 0],
    np.index_exp[-2:-7:-2, 0, 0],
    np.index_exp[-2:-7:-3, 0, 0],
    np.index_exp[-2:-7:-4, 0, 0],
    np.index_exp[-1::-4, 0, 0],
    np.index_exp[-6:-4, 0, 0],
    np.index_exp[2:4, 0:4, 0:4],
    np.index_exp[..., 2:],
    np.index_exp[..., -18:-16],
    np.index_exp[..., -19:-17],
    np.index_exp[..., -17:-19],
    np.index_exp[..., -19::2],
    np.index_exp[2:4, 0:10, 0:10],
    np.index_exp[2:4, 7:10, 7:10],
    np.index_exp[2, 0, -2],
    np.index_exp[2:5, :6, :],
    np.index_exp[2:5, 0:4, -10:-21:-1],
    np.index_exp[2:5, :, -10:-21:-1],
    np.index_exp[2:5, 0],
    np.index_exp[5, 5, 5],
    np.index_exp[5, -11:-9, 0:4],
    np.index_exp[5, -9:-11:-1, 0:4],
    np.index_exp[5, -2:-5:-1, 0:4],
    np.index_exp[5, -10, 0:4],
    np.index_exp[-5, -10, 0:4],
    np.index_exp[-5, -10, -10:-21:-1],
    np.index_exp[-5, 0, -10:-21:-1],
    np.index_exp[:, :, -10:-21:-2],
    np.index_exp[:, :, 0:10:2],
    np.index_exp[:, 0:10:2, 1:10:2],
    np.index_exp[:, 0, 1:10:2],
    np.index_exp[5],
    np.index_exp[:5],
    np.index_exp[:50],
    np.index_exp[:50, :50, :50],
    np.index_exp[:, 0:0:1, :],
    np.index_exp[..., 0:0:1],
]


@ddt
class TestDCIMGFILE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        f = DCIMGFile()

        f._sess_header = np.zeros(1, dtype=f.SESS_HDR_DTYPE)
        f._sess_header['nfrms'][0] = 10
        f._sess_header['ysize'][0] = 20
        f._sess_header['xsize'][0] = 20
        f._sess_header['byte_depth'][0] = 2
        f.mma = np.arange(np.prod(f.shape), dtype=np.uint16).reshape(f.shape)

        f._4px = (65535 - np.arange(f.nfrms * 4)).reshape((f.nfrms, 4))

        cls.f = f

        a_ok = np.copy(f.mma)
        a_ok[:, 0, 0:4] = f._4px

        cls.a_ok = a_ok

    @data(*test_vectors)
    def testGetItem(self, value):
        self.f.first_4px_correction_enabled = True
        a = self.f[value]
        if a.size == 0:
            self.assertEqual(a.size, self.a_ok[value].size)
        else:
            self.assertEqual(np.array_equal(self.a_ok[value], a), True)

    def testZSlice(self):
        a = self.f.zslice(3, 5)
        self.assertEqual(np.array_equal(self.a_ok[3:5], a), True)

    def testWhole(self):
        a = self.f.whole()
        self.assertEqual(np.array_equal(self.a_ok, a), True)


if __name__ == '__main__':
    unittest.main()
