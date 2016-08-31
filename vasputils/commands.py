import os
import copy
import argparse

from vasputils import core

class BaseCommand(object):
    def __init__(self):
        self.args = None

    def add_arguments(self):
        raise NotImplemented

    def __call__(self):
        self.parser = argparse.ArgumentParser()
        self.add_arguments()

        self.args = self.parser.parse_args()

        self.process_arguments()

    def process_arguments(self, args):
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



