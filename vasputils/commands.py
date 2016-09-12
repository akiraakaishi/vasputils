import os
import sys
import copy
import argparse

import numpy

from vasputils import core
from vasputils.vasprun import from_file


class BaseCommand(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_arguments()

        self.args = None

    def add_arguments(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        self.args = self.parser.parse_args(*args, **kwargs)
        self.process_arguments()

    def process_arguments(self):
        pass


class ChgSplit(BaseCommand):
    def add_arguments(self):
        self.parser.add_argument('CHGCAR', nargs='?', type=argparse.FileType(), help="CHG file. (default: './CHG')")
        self.parser.add_argument('--diff', action='store_true', default=False, help='output charge difference.')

    def process_arguments(self):
        f = self.args.CHGCAR
        if f is None:
            try:
                f = argparse.FileType()('CHG')
            except argparse.ArgumentTypeError as e:
                self.parser.error(e.message)
        chg = core.Chg.parse(f)

        if len(chg.charges) < 2:
            self.parser.error('no spin calculation')

        chg_new = core.Chg(comment=chg.comment, scale=chg.scale, matrix=chg.matrix.copy())
        chg_new.atoms = copy.deepcopy(chg.atoms)
        chg_new.dimension = chg.dimension
        if self.args.diff:
            chg_new.charges = [chg.charges[1]]

            fname = os.path.join(os.path.dirname(f.name), 'CHGCAR_diff')
            with open(fname, mode='w') as f_new:
                chg_new.write(f_new, direct=True)
        else:
            chg_new.charges = [chg.charges[0] - chg.charges[1]]
            fname = os.path.join(os.path.dirname(f.name), 'CHGCAR_0')
            with open(fname, mode='w') as f_new:
                chg_new.write(f_new, direct=True)

            chg_new.charges = [chg.charges[1] - chg.charges[0]]
            fname = os.path.join(os.path.dirname(f.name), 'CHGCAR_1')
            with open(fname, mode='w') as f_new:
                chg_new.write(f_new, direct=True)

chg_split = ChgSplit()


class RangeType(object):
    sep = ':'

    def __init__(self, t):
        self.type = t

    def __call__(self, string):
        split = string.split(self.sep)

        if len(split) > 2:
            raise argparse.ArgumentTypeError('invalid expression.')

        try:
            if len(split) == 1:
                start = self.type(split[0])
                end = start + 1
            else:
                start, end = split
                start = self.type(start) if start else None
                end = self.type(end) if end else None
        except ValueError as e:
            raise argparse.ArgumentTypeError('invalid expression: ' + e.message)
        return start, end


band_type = RangeType(int)
energy_type = RangeType(float)


class VasprunCommand(BaseCommand):
    def add_arguments(self):
        subparser = self.parser.add_subparsers(help='subparser', dest='command')
        self.parser.add_argument('--file', type=argparse.FileType(), help="vasprun.xml file. (default: './vasprun.xml')")

        eigenval_parser = subparser.add_parser('eigenval')
        eigenval_parser.add_argument('--energy', type=energy_type, default=(None, None), help="energy range separated by ':'. ")
        eigenval_parser.add_argument('--band', type=band_type, default=(None, None), help="band numbers separated by ':'. (ommitted if energy range is specified.)")

        dos_parser = subparser.add_parser('dos')
        dos_parser.add_argument('--integrate', action='store_true', default=False, help='')

    def process_arguments(self):
        f = self.args.file
        if f is None:
            try:
                f = argparse.FileType()('vasprun.xml')
            except argparse.ArgumentTypeError as e:
                self.parser.error(e.message)
        modeling = from_file(f)

        if self.args.command == 'eigenval':
            self.process_eigenval(modeling)
        elif self.args.command == 'dos':
            self.process_dos(modeling)

    def process_dos(self, modeling):
        calculation = modeling.calculations[-1]
        dos = calculation.dos

        total = numpy.array(dos.total.data)
        energy = total[0].take(0, axis=1) - dos.efermi
        energy = energy.reshape((energy.shape[0], 1))

        index = 2 if self.args.integrate else 1
        data = total.swapaxes(0, 2).take(index, axis=0)
        d = numpy.hstack((energy, data))
        numpy.savetxt(sys.stdout, d)

    def process_eigenval(self, modeling):
        kpoints = numpy.array(modeling.kpoints.kpointlist)
        k_diff = kpoints - numpy.concatenate([kpoints[:1], kpoints[:-1]])
        k_axis = numpy.cumsum(numpy.sqrt(numpy.power(k_diff, 2).sum(axis=1)))
        k_axis = k_axis.reshape((k_axis.shape[0], 1))

        calculation = modeling.calculations[-1]
        efermi = calculation.dos.efermi

        data = calculation.eigenvalues.array.data
        data = numpy.array(data).swapaxes(0, 2)

        bands = data.take(0, axis=3) - efermi
        num_bands = len(bands)

        start, end = self.args.band
        start = start or 0
        end = end or num_bands
        if start < 0 or end < 0:
            occupations = data.take(1, axis=3)
            zero = (numpy.average(occupations, axis=1) > 0.001).sum()
            if start is not None:
                start = start + zero
            if end is not None:
                end = end + zero

        e_min, e_max = self.args.energy
        if (e_min is not None) or (e_max is not None):
            start, end = 0, num_bands
            if e_min is not None:
                start = (bands.max(axis=1) < e_min).sum()

            if e_max is not None:
                end = (bands.min(axis=1) < e_max).sum()

        start = max(start, 0)
        end = min(end, len(bands))
        for band in bands[start:end]:
            d = numpy.hstack((k_axis, band))
            numpy.savetxt(sys.stdout, d)
            sys.stdout.write('\n')


vasprun = VasprunCommand()
