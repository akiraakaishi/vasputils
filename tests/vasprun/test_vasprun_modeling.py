import os

from nose.tools import eq_

from vasputils.vasprun import from_file


def test_modeling():
    vasprun_01_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'vasprun_01.xml')
    with open(vasprun_01_filename) as f:
        modeling = from_file(f)
        calculation = modeling.calculations[-1]
        data = calculation.eigenvalues.array.data[0]
        eq_(data[0][0], [-22.4951, 1.0000])
