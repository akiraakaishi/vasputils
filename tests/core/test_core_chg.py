import os
import StringIO

from vasputils.core import Chg, Chgcar


def test_chg():
    CHG_01_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'CHG_01')
    with open(CHG_01_filename) as f:
        chg = Chg.parse(f)

        buf = StringIO.StringIO()
        chg.write(buf)


def test_chgcar():
    CHGCAR_01_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'CHGCAR_01')
    with open(CHGCAR_01_filename) as f:
        import warnings
        # suppress warnings when loading empty string by `numpy.loadtxt`
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            chgcar = Chgcar.parse(f)

        buf = StringIO.StringIO()
        chgcar.write(buf)
