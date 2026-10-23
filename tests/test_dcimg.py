import os
import tempfile
import unittest
from ddt import ddt, data

import numpy as np

from dcimg import DCIMGFile


test_vectors = [
    np.index_exp[...],
    np.index_exp[..., -1],
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
    np.index_exp[0, 0, 0],
    np.index_exp[0, 0, 2],
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
    np.index_exp[5, 1022, 0],
    np.index_exp[5, 1022, :4],
    np.index_exp[5, :1023, :4],
    np.index_exp[5, 1000:1023, :4],
    np.index_exp[5, 1000:1030, :4],
    np.index_exp[:, ::2, :4],
    np.index_exp[:, 1::2, :4],
    np.index_exp[5, 1030:1000:-1, :4],
    np.index_exp[5, 1023:1000:-2, :4],
    np.index_exp[5, ::-1, :4],
    np.index_exp[5, 1021:1030:3, :4],
    np.index_exp[-3:, :50, :4],
    np.index_exp[::-2, 0, ::-1],
    np.index_exp[np.int64(5), np.int64(0), np.int64(1)],
    np.index_exp[np.int64(5), 1023, :8],
]


class DCIMGFileOverride4px(DCIMGFile):
    @property
    def _has_4px_data(self):
        return True

    def compute_target_line(self):
        self._target_line = 0


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

        f = DCIMGFileOverride4px()
        f._sess_header = _sess_header
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
        _sess_header = np.zeros(1, dtype=DCIMGFile.NEW_SESSION_HEADER_DTYPE)
        _sess_header['nfrms'][0] = nfrms
        _sess_header['ysize'][0] = 2048
        _sess_header['xsize'][0] = 2048
        _sess_header['byte_depth'][0] = 2
        _sess_header['frame_footer_size'][0] = 32
        f._sess_header = _sess_header
        _file_header = np.zeros(1, dtype=DCIMGFile.FILE_HDR_DTYPE)
        _file_header['format_version'] = 0x1000000
        f._file_header = _file_header
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
                self.assertTrue(np.array_equal(a_ok[value], a))

    def testZSlice(self):
        for f, a_ok in self.test_config:
            a = f.zslice(3)
            self.assertTrue(np.array_equal(a_ok[:3], a))

            a = f.zslice(3, 5)
            self.assertTrue(np.array_equal(a_ok[3:5], a))

            a = f.zslice(3, 8, 2)
            self.assertTrue(np.array_equal(a_ok[3:8:2], a))

            a = f.zslice(-8, -3, 2)
            self.assertTrue(np.array_equal(a_ok[-8:-3:2], a))

            a = f.zslice(-8, -3, -2)
            self.assertTrue(np.array_equal(a_ok[-8:-3:-2], a))
            self.assertEqual(0, a.shape[0])

    def testWhole(self):
        for f, a_ok in self.test_config:
            a = f.whole()
            self.assertTrue(np.array_equal(a_ok, a))

    def testArgsToSlice(self):
        self.assertEqual(slice(42), DCIMGFile._args_to_slice(42))
        self.assertEqual(slice(10, 42), DCIMGFile._args_to_slice(10, 42))
        self.assertEqual(slice(10, 42, 5), DCIMGFile._args_to_slice(10, 42, 5))

        self.assertEqual(slice(42), DCIMGFile._args_to_slice(None, 42, None))


def _write_dcimg(path, frames, fmt, px4=None, px4_row=0):
    """Write a minimal DCIMG file holding `frames` (nfrms, ysize, xsize)."""
    nfrms, ysize, xsize = frames.shape
    header_size = 232
    offset_to_data = 1024
    bytes_per_img = ysize * xsize * 2
    fh = np.zeros(1, DCIMGFile.FILE_HDR_DTYPE)
    fh['file_format'] = b'DCIMG'
    fh['nsess'] = 1
    fh['nfrms'] = nfrms
    fh['header_size'] = header_size
    if fmt == DCIMGFile.FMT_OLD:
        fh['format_version'] = 0x7
        sh = np.zeros(1, DCIMGFile.SESS_HDR_DTYPE)
        sh['session_data_size'] = offset_to_data + bytes_per_img * nfrms
        frame_footer = 0
    else:
        fh['format_version'] = 0x1000000
        sh = np.zeros(1, DCIMGFile.NEW_SESSION_HEADER_DTYPE)
        frame_footer = 16
        sh['frame_footer_size'] = frame_footer
    sh['nfrms'] = nfrms
    sh['byte_depth'] = 2
    sh['xsize'] = xsize
    sh['ysize'] = ysize
    sh['bytes_per_row'] = xsize * 2
    sh['bytes_per_img'] = bytes_per_img
    sh['offset_to_data'] = offset_to_data

    data_start = header_size + offset_to_data
    data_end = data_start + (bytes_per_img + frame_footer) * nfrms
    buf = bytearray(data_end + 4096)
    buf[:fh.nbytes] = fh.tobytes()
    buf[header_size:header_size + sh.nbytes] = sh.tobytes()
    if fmt == DCIMGFile.FMT_NEW:
        crop = np.zeros(1, DCIMGFile.NEW_CROP_INFO)
        crop['xsize'] = xsize
        crop['ysize'] = ysize
        i = header_size + 712
        buf[i:i + crop.nbytes] = crop.tobytes()
    else:
        footer = np.zeros(1, DCIMGFile.SESSION_FOOTER_DTYPE)
        footer['offset_to_2nd_struct'] = footer.nbytes
        footer2 = np.zeros(1, DCIMGFile.SESSION_FOOTER2_DTYPE)
        if px4 is not None:
            offset_to_4px = 512 + 12 * nfrms
            footer['footer_size'] = offset_to_4px + 8 * nfrms
            footer2['offset_to_4px'] = offset_to_4px
            footer2['4px_offset_in_frame'] = px4_row * xsize * 2
            i = data_end + offset_to_4px
            buf[i:i + px4.nbytes] = px4.astype('<u2').tobytes()
        buf[data_end:data_end + footer.nbytes] = footer.tobytes()
        i = data_end + footer.nbytes
        buf[i:i + footer2.nbytes] = footer2.tobytes()
    for z in range(nfrms):
        i = data_start + z * (bytes_per_img + frame_footer)
        buf[i:i + bytes_per_img] = frames[z].astype('<u2').tobytes()
    with open(path, 'wb') as fp:
        fp.write(buf)


class TestDCIMGFileOnDisk(unittest.TestCase):

    def testOpenAndIndex(self):
        frames = np.arange(3 * 6 * 16, dtype=np.uint16).reshape((3, 6, 16))
        for fmt in (DCIMGFile.FMT_OLD, DCIMGFile.FMT_NEW):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, 'test.dcimg')
                _write_dcimg(path, frames, fmt)
                f = DCIMGFile(path)
                self.assertEqual(f.shape, frames.shape)
                self.assertFalse(f._has_4px_data)
                self.assertTrue(np.array_equal(f.whole(), frames))
                self.assertTrue(np.array_equal(f[-1], frames[-1]))
                self.assertTrue(np.array_equal(f[..., -5:], frames[..., -5:]))
                self.assertTrue(np.array_equal(f[1, -2, ::-1], frames[1, -2, ::-1]))
                f.close()

    def testOld4pxOnTargetLine(self):
        # the 4px data replaces the first 4 pixels of row 3, not of row 0
        frames = np.arange(3 * 6 * 16, dtype=np.uint16).reshape((3, 6, 16))
        px4 = 60000 + np.arange(3 * 4, dtype=np.uint16).reshape((3, 4))
        a_ok = frames.copy()
        a_ok[:, 3, :4] = px4
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'test.dcimg')
            _write_dcimg(path, frames, DCIMGFile.FMT_OLD, px4, px4_row=3)
            f = DCIMGFile(path)
            self.assertTrue(f._has_4px_data)
            self.assertEqual(f._target_line, 3)
            self.assertTrue(np.array_equal(f.whole(), a_ok))
            for idx in [np.index_exp[1, 3, 0], np.index_exp[1, 2, 0],
                        np.index_exp[:, 0:3, :4], np.index_exp[:, 2:5, :4],
                        np.index_exp[:, :4, :2], np.index_exp[:, ::-1, :],
                        np.index_exp[2, 3]]:
                self.assertTrue(np.array_equal(f[idx], a_ok[idx]), idx)
            f.close()


if __name__ == '__main__':
    unittest.main()
