import argparse
import unittest

from vasputils.commands import RangeType


class RangeTypeTest(unittest.TestCase):
    def setUp(self):
        self.int_range_type = RangeType(int)
        self.float_range_type = RangeType(float)

    def test_int_range(self):
        range_type = self.int_range_type
        self.assertEqual(range_type('1:3'), (1, 3))
        self.assertEqual(range_type('-1:'), (-1, None))
        self.assertEqual(range_type(':3'), (None, 3))
        self.assertEqual(range_type('3'), (3, 4))

        with self.assertRaises(argparse.ArgumentTypeError):
            range_type('too:many:commas')

    def test_float_range(self):
        range_type = self.float_range_type
        self.assertTrue(range_type('1.0:'), (1.0, None))

        with self.assertRaises(argparse.ArgumentTypeError):
            range_type('some string')
