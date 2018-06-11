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
    np.index_exp[:3000, :3000, :3000],
    np.index_exp[:, 0:0:1, :],
    np.index_exp[..., 0:0:1],
]


@ddt
class TestDCIMGFILE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nfrms = 10
        _sess_header = np.zeros(1, dtype=DCIMGFile.SESS_HDR_DTYPE)
        _sess_header['nfrms'][0] = nfrms
        _sess_header['ysize'][0] = 2048
        _sess_header['xsize'][0] = 2048
        _sess_header['byte_depth'][0] = 2

        _4px = (65000 - np.arange(nfrms * 4)).reshape((nfrms, 4))

        f = DCIMGFile()
        f._sess_header = np.copy(_sess_header)
        f.mma = np.arange(np.prod(f.shape), dtype=np.uint16).reshape(f.shape)
        f.mma.flags.writeable = False
        f.deep_copy_enabled = True
        f.first_4px_correction_enabled = True
        f.fmt_version = DCIMGFile.FMT_OLD
        f._4px = np.copy(_4px)
        f._4px.flags.writeable = False
        f.compute_target_line()

        cls.f = f

        f = DCIMGFile()
        f._sess_header = np.copy(_sess_header)
        f.mma = np.arange(np.prod(f.shape), dtype=np.uint16).reshape(f.shape)
        f.mma.flags.writeable = False
        f.deep_copy_enabled = True
        f.first_4px_correction_enabled = True
        f.fmt_version = DCIMGFile.FMT_NEW
        f._4px = np.copy(_4px)
        f._4px.flags.writeable = False
        f.compute_target_line()

        cls.f_new = f

        a_ok = np.copy(cls.f.mma)
        a_ok[:, 0, 0:4] = f._4px
        cls.a_ok = np.copy(a_ok)
        cls.a_ok.flags.writeable = False

        a_ok = np.copy(cls.f_new.mma)
        a_ok[:, 1023, 0:4] = cls.f_new._4px
        cls.a_ok_new = np.copy(a_ok)
        cls.a_ok_new.flags.writeable = False

        cls.test_config = list(zip([cls.f, cls.f_new],
                                   [cls.a_ok, cls.a_ok_new]))

    @data(*test_vectors)
    def testGetItem(self, value):
        for f, a_ok in self.test_config:
            a = f[value]
            if a.size == 0:
                self.assertEqual(a.size, a_ok[value].size)
            else:
                self.assertEqual(np.array_equal(a_ok[value], a), True)

    def testZSlice(self):
        for f, a_ok in self.test_config:
            a = f.zslice(3, 5)
            self.assertEqual(np.array_equal(a_ok[3:5], a), True)

    def testWhole(self):
        for f, a_ok in self.test_config:
            a = f.whole()
            self.assertEqual(np.array_equal(a_ok, a), True)


if __name__ == '__main__':
    unittest.main()
