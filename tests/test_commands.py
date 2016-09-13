import os
import sys
from contextlib import contextmanager
from cStringIO import StringIO

import argparse
import unittest

from vasputils.commands import RangeType, vasprun


@contextmanager
def capture(command, *args, **kwargs):
  out, sys.stdout = sys.stdout, StringIO()
  try:
    command(*args, **kwargs)
    sys.stdout.seek(0)
    yield sys.stdout.read()
  finally:
    sys.stdout = out



class VasprunTest(unittest.TestCase):
    def setUp(self):
        self.vasprun_01_filename = os.path.join(os.path.dirname(__file__), 'vasprun', 'fixtures', 'vasprun_01.xml')
        self.file_arg = '--file={}'.format(self.vasprun_01_filename)

    def vasprun(self, args):
        try:
            vasprun(args)
        except SystemExit as e:
            assert e.code == 0

    def test_dos(self):
        args = (self.file_arg, 'dos')
        self.vasprun(args)

    def test_eigenval(self):
        args = (self.file_arg, 'eigenval')
        self.vasprun(args)
        args = (self.file_arg, 'eigenval', '--band=-5:5')
        self.vasprun(args)
        args = (self.file_arg, 'eigenval', '--energy=-5.0:5.0')
        self.vasprun(args)


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
