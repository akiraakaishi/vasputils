import StringIO
import operator
import re

import numpy

def read_array(f, n=1, dtype=float, ndmin=1):
    ''' read n lines from file f and convert these lines to numpy.array.
    '''
    buf = StringIO.StringIO()
    for i in range(n):
        buf.write(f.readline())
    buf.seek(0)
    return numpy.loadtxt(buf, dtype=dtype, ndmin=ndmin)

class Element(object):
    table = {
        # Radius: [http://crystalmaker.com/support/tutorials/crystalmaker/atomic-radii/index.html]
        # Color: [http://jmol.sourceforge.net/jscolors/]
        # 'Symbol': (Number, Radius, Color)
        'H': (1, 0.53, 'FFFFFF'),
        'He': (2, 0.31, 'D9FFFF'),
        'Li': (3, 1.67, 'CC80FF'),
        'Be': (4, 1.12, 'C2FF00'),
        'B': (5, 0.87, 'FFB5B5'),
        'C': (6, 0.67, '909090'),
        'N': (7, 0.56, '3050F8'),
        'O': (8, 0.48, 'FF0D0D'),
        'F': (9, 0.42, '90E050'),
        'Ne': (10, 0.38, 'B3E3F5'),
        'Na': (11, 1.90, 'AB5CF2'),
        'Mg': (12, 1.45, '8AFF00'),
        'Al': (13, 1.18, 'BFA6A6'),
        'Si': (14, 1.11, 'F0C8A0'),
        'P': (15, 0.98, 'FF8000'),
        'S': (16, 0.88, 'FFFF30'),
        'Cl': (17, 0.79, '1FF01F'),
        'Ar': (18, 0.71, '80D1E3'),
        'K': (19, 2.43, '8F40D4'),
        'Ca': (20, 1.94, '3DFF00'),
        'Sc': (21, 1.84, 'E6E6E6'),
        'Ti': (22, 1.76, 'BFC2C7'),
        'V': (23, 1.71, 'A6A6AB'),
        'Cr': (24, 1.66, '8A99C7'),
        'Mn': (25, 1.61, '9C7AC7'),
        'Fe': (26, 1.56, 'E06633'),
        'Co': (27, 1.52, 'F090A0'),
        'Ni': (28, 1.49, '50D050'),
        'Cu': (29, 1.45, 'C88033'),
        'Zn': (30, 1.42, '7D80B0'),
        'Ga': (31, 1.36, 'C28F8F'),
        'Ge': (32, 1.25, '668F8F'),
        'As': (33, 1.14, 'BD80E3'),
        'Se': (34, 1.03, 'FFA100'),
        'Br': (35, 0.94, 'A62929'),
        'Kr': (36, 0.88, '5CB8D1'),
    }
    def __init__(self, symbol):
        if symbol in self.table:
            number, radius, color = self.table[symbol]
        else:
            number, radius, color = (0, 1, '000000')

        self.symbol = symbol
        self.number = number
        self.radius = radius
        self.color = color

class Atom(object):
    def __init__(self, name):
        self.name = name
        self.coordinates = []

        self.element = Element(name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def counts(self):
        return len(self.coordinates)


class Basecar(object):
    @classmethod
    def parse(cls, f):
        ins = cls()
        ins._parse(f)
        return ins

    def _parse(self, f):
        raise NotImplemented

    def write(self, f):
        raise NotImplemented


class Contcar(Basecar):
    def __init__(self, comment='', scale=None, matrix=None):
        self.comment = comment
        self.scale = scale
        self.matrix = matrix

        self.atoms = []

    def _parse_header(self, f):
        comment = f.readline().rstrip()
        scale = float(f.readline())
        matrix = numpy.matrix(read_array(f, 3))

        self.comment = comment
        self.scale = scale
        self.matrix = matrix

    def _parse(self, f):
        self._parse_header(f)
        self._parse_body(f)

    def _parse_body(self, f):
        atoms = map(Atom, read_array(f, dtype=str))
        counts = read_array(f, dtype=int)

        is_cartesian = f.readline().lstrip().lower()[0] in 'ck'

        for atom, count in zip(atoms, counts):
            for c in range(count):
                coordinate = read_array(f)
                # store data always in Cartesian coordinate.
                if not is_cartesian:
                    coordinate = coordinate.dot(self.matrix)
                atom.coordinates.append(coordinate)
            self.atoms.append(atom)

    def atom_counts(self):
        return sum([atom.counts for atom in self.atoms])

    def write(self, f, direct=True):
        lines = [self.comment, '   {:.14f}'.format(self.scale), '']
        f.write('\n'.join(lines))
        numpy.savetxt(f, self.matrix, fmt='   % .6f')
        lines = [''.join(['   {:s}'.format(atom.name) for atom in self.atoms]), ''.join(['    {:d}'.format(atom.counts) for atom in self.atoms]), 'Direct' if direct else 'Cartesian', '']
        f.write('\n'.join(lines))

        for atom in self.atoms:
            for coordinate in atom.coordinates:
                data = coordinate.dot(self.matrix.I) if direct else coordinate
                numpy.savetxt(f, data, fmt=' % .6f')


class Poscar(Contcar):
    pass

def read_sequence(f, counts, columns=5):
    data = read_array(f, n=counts / columns).flatten()
    if counts % columns != 0:
        data = numpy.append(data, read_array(f))
    return data

class Chg(Contcar):
    columns = 10
    chg_fmt = '% .5E'
    def __init__(self, *args, **kwargs):
        super(Chg, self).__init__(*args, **kwargs)

        self.dimension = ()
        self.charges = []

    def _parse(self, f):
        super(Chg, self)._parse(f)
        self._parse_charge(f)

    def _parse_charge(self, f):

        f.readline() # skip a blank line
        self._read_charge(f)

        try:
            self._read_charge(f)
        except StopIteration:
            pass

    def _read_charge(self, f):
        dimension = read_array(f, dtype=int)
        self.dimension = tuple(dimension.tolist())

        counts = reduce(operator.mul, self.dimension)
        charge = read_sequence(f, counts, columns=self.columns)

        charge = charge.reshape(self.dimension)
        self.charges.append(charge)


    def write(self, f, direct=True):
        Contcar.write(self, f, direct=direct)

        f.write('\n')
        numpy.savetxt(f, [self.dimension], fmt='   %d')
        
        charge = self.charges[0].flatten()
        columns = self.columns
        row = (charge.size / columns)
        data = charge[:row * columns].reshape((row, columns))
        numpy.savetxt(f, data, fmt=self.chg_fmt)
        numpy.savetxt(f, [charge[row * columns:]], fmt=self.chg_fmt)

class Chgcar(Chg):
    columns = 5
    chg_fmt = '% .11E'
    def _read_aug(self, f):
        pattern = re.compile(r'^augmentation occupancies\s+(?P<atom_number>\d+)\s+(?P<data_count>\d+)$')
        for i in range(self.atom_counts()):
            m = pattern.match(f.readline())
            if m:
                assert int(m.group('atom_number')) == i + 1
                data_count = int(m.group('data_count'))
                if data_count:
                    augmentation_occupancy = read_sequence(f, data_count)
                else:
                    f.readline() # skip a blank line

    def _parse_charge(self, f):

        f.readline() # skip a blank line
        self._read_charge(f)
        self._read_aug(f)

        try:
            read_sequence(f, self.atom_counts())
            self._read_charge(f)
            self._read_aug(f)
        except StopIteration:
            pass




