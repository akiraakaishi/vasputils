import unittest

from vasputils.vasprun import from_string


class VectorTest(unittest.TestCase):
    def test_parse(self):
        v = from_string('<v>0.0 1.0 1.2</v>')
        self.assertTrue(v.value, [0.0, 1.0, 1.2])
