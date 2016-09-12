import re
import collections
from xml.etree import ElementTree


class VaspParseError(Exception):
    pass


def from_element(element):
    """

    >>> element = ElementTree.fromstring('<i name="bar">foo</i>')
    >>> from_element(element)
    Item(bar, foo)
    """

    if element.tag not in MetaEntry.tags:
        raise VaspParseError
    cls = MetaEntry.tags[element.tag]
    return cls.from_element(element)


def from_string(xml_string):
    try:
        element = ElementTree.fromstring(xml_string)
    except ElementTree.ParseError as e:
        raise VaspParseError(e)

    return from_element(element)


def from_file(source):
    try:
        element_tree = ElementTree.parse(source)
    except ElementTree.ParseError as e:
        raise VaspParseError(e)

    return from_element(element_tree.getroot())


def from_stringlist(sequence):
    try:
        element = ElementTree.fromstringlist(sequence)
    except ElementTree.ParseError as e:
        raise VaspParseError(e)

    return from_element(element)


class MetaEntry(type):
    tags = {}

    def __new__(cls, name, bases, attrs):
        new_class = super(MetaEntry, cls).__new__(cls, name, bases, attrs)
        tag = attrs.get('tag', None)
        if tag:
            cls.tags[tag] = new_class
        return new_class


class BaseEntry(object):
    """

    >>> entry = BaseEntry.from_element(None)
    >>> class Some(BaseEntry): tag = 'some_tag'
    >>> Some.from_element(ElementTree.fromstring('<wrong_tag name="bar">foo</wrong_tag>'))
    Traceback (most recent call last):
      ...
    ValueError: incorrect tag element
    """

    __metaclass__ = MetaEntry
    tag = None
    element = None

    @classmethod
    def from_element(cls, element):
        if cls.tag and cls.tag != element.tag:
            raise ValueError('incorrect tag element')

        ins = cls._instantiate(element)
        ins.element = element
        return ins

    @classmethod
    def from_string(cls, string):
        element = ElementTree.fromstring(string)
        return cls.from_element(element)

    @classmethod
    def _instantiate(cls, element):
        return cls()


class Entry(BaseEntry):
    """

    >>> from xml.etree import ElementTree
    >>> element = ElementTree.fromstring('<i name="bar">foo</i>')
    >>> e = Entry.from_element(element)
    >>> e
    Entry(bar, foo)
    """

    def __init__(self, name=None, value=None):
        self.name = name
        self.raw_value = value
        if value is None:
            self._value = value

    def __repr__(self):
        return '{classname}({name}, {value})'.format(classname=self.__class__.__name__, name=self.name, value=self.value)

    def __str__(self):
        return '{name}={value}'.format(name=self.name, value=self.value)

    @property
    def value(self):
        if hasattr(self, '_value'):
            return self._value

        self._value = self._evaluate()
        return self._value

    def _evaluate(self):
        return self.raw_value

    @classmethod
    def _instantiate(cls, element):
        name = element.get('name', None)
        value = element.text
        return cls(value=value, name=name)


class TypedEntry(Entry):
    def __init__(self, name, value, type_=None):
        super(TypedEntry, self).__init__(name=name, value=value)
        self.type = type_

    @classmethod
    def _instantiate(cls, element):
        name = element.get('name', None)
        type_ = element.get('type', None)
        value = element.text
        return cls(value=value, name=name, type_=type_)


class Item(TypedEntry):
    """
    Item

    >>> Item('bar', ' foo ', type_='string')
    Item(bar, foo)
    >>> Item('day', '  1', type_='int')
    Item(day, 1)
    >>> Item('happy', '  T ', type_='logical')
    Item(happy, True)
    >>> Item('lucky', '  F ', type_='logical')
    Item(lucky, False)
    >>> Item('foo', ' True', type_='logical')
    Traceback (most recent call last):
      ...
    ValueError: value must be 'T' or 'F' for logical type parameter
    >>> print Item('ratio', '  10.000 ')
    ratio=10.0
    >>> from xml.etree import ElementTree
    >>> element = ElementTree.fromstring('<i name="day" type="int">2</i>')
    >>> Item.from_element(element)
    Item(day, 2)
    """

    tag = 'i'

    def _evaluate(self):
        if self.type == 'int':
            return int(self.raw_value)
        elif self.type == 'string':
            return self.raw_value.strip()
        elif self.type == 'logical':
            T_or_F = self.raw_value.strip()
            if T_or_F == 'T':
                return True
            elif T_or_F == 'F':
                return False
            else:
                raise ValueError("value must be 'T' or 'F' for logical type parameter")
        else:
            try:
                return float(self.raw_value)
            except ValueError:
                pass

        return self.raw_value


class Vector(TypedEntry):
    """

    >>> vector = Vector('0.1  0.2', name='bar')
    >>> vector
    Vector(bar, [0.1, 0.2])
    >>> vector.value
    [0.1, 0.2]
    >>> vector = Vector(' 1  2 ', type_='int')
    >>> vector.value
    [1, 2]
    >>> Vector.from_string('<v>0.1 0.2</v>').value
    [0.1, 0.2]
    """

    tag = 'v'

    def __init__(self, value, name=None, type_=None):
        super(Vector, self).__init__(name=name, value=value)
        self.type = type_

    def _evaluate(self):
        dtype = float
        if self.type == 'int':
            dtype = int
        sep = ' '
        squeezed = re.sub(r'\s+', sep, self.raw_value)
        stripped = squeezed.strip()
        values = stripped.split(sep)
        return map(dtype, values)


class Time(Vector):
    tag = 'time'


class VectorArray(Entry):
    """

    >>> array = VectorArray.from_string('<varray><v>0.0 0.1</v><v>0.4 0.3</v></varray>')
    >>> array.value
    [[0.0, 0.1], [0.4, 0.3]]
    """

    tag = 'varray'

    def __init__(self, name=None):
        super(VectorArray, self).__init__(name=name)
        self.array = None

    def add(self, vector):
        if self.array is None:
            self.array = []
        self.array.append(vector.value)
        if hasattr(self, '_value'):
            delattr(self, '_value')

    def _evaluate(self):
        return self.array

    @classmethod
    def _instantiate(cls, element):
        ins = cls(name=element.get('name', None))

        for e in element.iter('v'):
            v = Vector.from_element(e)
            ins.add(v)

        return ins


class Array(Entry):
    """
    >>> array = Array(['spin', 'kpoint', 'band'], ['eigene', ('occ', float)])
    """

    tag = 'array'

    def __init__(self, dimensions, fields, name=None):
        super(Array, self).__init__(name=name)
        if len(dimensions) < 1:
            raise ValueError('at least 1 dimension required')

        self.dimensions = dimensions

        dtype = []
        for field in fields:
            if isinstance(field, (tuple, list)):
                name, type_ = field[:2]
            else:
                name = field
                type_ = float
            dtype.append((name, type_))

        self.dtype = dtype

        self.set = collections.OrderedDict()
        self._data = None

    @property
    def data(self):
        if self._data is not None:
            return self._data

        def _get_data(s):
            if isinstance(s, dict):
                return [_get_data(d) for d in s.itervalues()]
            return s

        self._data = _get_data(self.set)[0]
        return self._data

    @classmethod
    def _instantiate(cls, element):
        dimensions = []
        for e in element.iter('dimension'):
            dimensions.append(e.text)

        fields = []
        for e in element.iter('field'):
            type_str = e.get('type', None)
            if type_str == 'int':
                type_ = int
            elif type_str == 'string':
                # type_ = object # store string data as object
                type_ = str
            else:
                type_ = float
            fields.append((e.text, type_))

        name = element.get('name', None)
        ins = cls(dimensions, fields, name=name)

        types = [dtype[1] for dtype in ins.dtype]

        def _extract_set(elem, set):
            if elem.tag in ('r', 'rc'):
                if elem.tag == 'r':
                    text = elem.text.strip()
                    raw_set = filter(None, text.split(' '))
                else:
                    raw_set = [c.text.strip() for c in elem.iter('c')]
                d = [t(raw_datum) for t, raw_datum in zip(types, raw_set)]

                index = len(set)
                set[index] = d
            elif elem.tag == 'set':
                comment = elem.get('comment', None)
                set[comment] = collections.OrderedDict()
                for e in elem:
                    _extract_set(e, set[comment])

        _extract_set(element.find('set'), ins.set)

        return ins


class MultipleEntry(BaseEntry):
    """

    >>> entry_set = MultipleEntry()
    >>> entry_set.add(Item(name='bar', value='foo', type_='string'))
    >>> entry_set.get('bar')
    'foo'
    """

    entry_tags = ()

    def __init__(self, entries=[]):
        super(MultipleEntry, self).__init__()

        self.entries = {}
        for entry in entries:
            self.add(entry)

    def add(self, entry):
        if self._validate_entry(entry):
            self.entries[entry.name] = entry

    def get(self, name):
        if name not in self.entries:
            return None

        entry = self.entries[name]
        return entry.value

    def _validate_entry(self, entry):
        if self.entry_tags:
            return entry.tag in self.entry_tags
        return True

    @classmethod
    def _instantiate(cls, element):
        ins = cls()
        for sub_element in element:
            if sub_element.tag in cls.entry_tags:
                entry = from_element(sub_element)
                ins.add(entry)
        return ins


class ItemSet(MultipleEntry):
    entry_tags = ('i', )

    def __init__(self, items=[]):
        super(ItemSet, self).__init__(entries=items)


class Energy(MultipleEntry):
    tag = 'energy'
    entry_tags = ('i', )


class Section(collections.OrderedDict):
    """

    >>> section = Section()
    >>> section.add_entry(None)
    >>> section.entries
    [None]
    """
    def __init__(self, *args, **kwargs):
        super(Section, self).__init__(*args, **kwargs)
        self.entries = []

    def add_entry(self, entry):
        self.entries.append(entry)


class Parameters(MultipleEntry):
    """

    >>> param = Parameters()
    >>> param.add(Item(name='bar', value='foo'))
    >>> param.add(Item(name='nested', value='value'), directory='/sub dir')
    >>> param.get('bar')
    'foo'
    >>> param = Parameters.from_string('<parameters><section name="some section"><i name="bar">0.0</i></section></parameters>')
    """

    entry_tags = ('i', 'v', )
    tag = 'parameters'

    def __init__(self):
        super(Parameters, self).__init__()
        self.section = Section()

    def add(self, item, directory='/'):
        section = self._get_section(directory)
        super(Parameters, self).add(item)
        section.add_entry(item)

    def _get_section(self, directory):
        if not directory.lstrip().startswith('/'):
            raise ValueError("directory must have leading '/'")

        names = directory.split('/')[1:]
        names = filter(None, names)  # remove empty entry

        section = self.section
        for name in names:
            if name not in section:
                section[name] = Section()
            section = section[name]
        return section

    @classmethod
    def _instantiate(cls, element):
        def _parse_parameter(elem, directory, parameter):
            if elem.tag in cls.entry_tags:
                entry = from_element(elem)
                parameter.add(entry, directory=directory)
            elif elem.tag == 'separator':
                name = elem.get('name')
                if directory.rstrip().endswith('/'):
                    directory = ''.join([directory, name])
                else:
                    directory = '/'.join([directory, name])
                for e in elem:
                    _parse_parameter(e, directory, parameter)

        ins = cls()
        for sub_element in element:
                _parse_parameter(sub_element, '/', ins)
        return ins


class Kpoints(BaseEntry):
    tag = 'kpoints'

    def __init__(self, kpointlist, weights, generation=None):
        super(Kpoints, self).__init__()
        self.kpointlist = kpointlist
        self.weights = weights
        self.generation = generation

    @classmethod
    def _instantiate(cls, element):
        generation = None
        generation_element = element.find('generation')
        if generation_element is not None:
            generation = ItemSet()

            param_value = generation_element.get('param', None)
            param = Item(name='param', value=param_value, type_='string')
            generation.add(param)

            for v in generation_element.iter('v'):
                generation.add(Vector.from_element(v))

        kpointlist_varray = from_element(element.find("varray[@name='kpointlist']"))
        weights_varray = from_element(element.find("varray[@name='weights']"))

        kpointlist = kpointlist_varray.value
        weights = weights_varray.value

        return cls(kpointlist, weights, generation)


class Atominfo(BaseEntry):
    tag = 'atominfo'

    def __init__(self, n_atoms, n_types, atoms, atomtypes):
        super(Atominfo, self).__init__()
        self.n_atoms = n_atoms
        self.n_types = n_types
        self.atoms = atoms
        self.atomtypes = atomtypes

    @classmethod
    def _instantiate(cls, element):
        n_atoms = int(element.find('atoms').text)
        n_types = int(element.find('types').text)

        atoms = Array.from_element(element.find("array[@name='atoms']"))
        atomtypes = Array.from_element(element.find("array[@name='atomtypes']"))

        return cls(n_atoms, n_types, atoms, atomtypes)


class Crystal(BaseEntry):
    tag = 'crystal'

    def __init__(self, basis, rec_basis, volume):
        super(Crystal, self).__init__()
        self.basis = basis
        self.rec_basis = rec_basis
        self.volume = volume

    @classmethod
    def _instantiate(cls, element):
        basis_varray = from_element(element.find("varray[@name='basis']"))
        rec_basis_varray = from_element(element.find("varray[@name='rec_basis']"))
        volume_i = Item.from_element(element.find("i[@name='volume']"))

        basis = basis_varray.value
        rec_basis = rec_basis_varray.value
        volume = volume_i.value

        return cls(basis, rec_basis, volume)


class Structure(BaseEntry):
    tag = 'structure'

    def __init__(self, crystal, positions):
        super(Structure, self).__init__()
        self.crystal = crystal
        self.positions = positions

    @classmethod
    def _instantiate(cls, element):
        crystal = Crystal.from_element(element.find('crystal'))
        positions_varray = from_element(element.find("varray[@name='positions']"))
        positions = positions_varray.value

        return cls(crystal, positions)


class SCStep(BaseEntry):
    tag = 'scstep'

    def __init__(self, energy, times):
        super(SCStep, self).__init__()
        self.energy = energy
        self.times = times

    @classmethod
    def _instantiate(cls, element):
        energy = Energy.from_element(element.find('energy'))
        times = []
        for time_element in element.iter('time'):
            time = Time.from_element(time_element)
            times.append(time)

        return cls(energy, times)


class Eigenvalues(BaseEntry):
    tag = 'eigenvalues'

    def __init__(self, array):
        super(Eigenvalues, self).__init__()
        self.array = array

    @classmethod
    def _instantiate(cls, element):
        array = Array.from_element(element.find('array'))
        return cls(array)


class DOS(BaseEntry):
    tag = 'dos'

    def __init__(self, efermi, total, partial=None):
        self.efermi = efermi
        self.total = total
        self.partial = partial

    @classmethod
    def _instantiate(cls, element):
        efermi_i = Item.from_element(element.find("i[@name='efermi']"))
        total_element = element.find('total')
        total_array = Array.from_element(total_element.find('array'))

        partial_element = element.find('partial')
        partial_array = None
        if partial_element is not None:
            partial_array = Array.from_element(partial_element.find('array'))

        return cls(efermi_i.value, total_array, partial_array)


class Projected(BaseEntry):
    tag = 'projected'

    def __init__(self, eigenvalues, array):
        self.eigenvalues = eigenvalues
        self.array = array

    @classmethod
    def _instantiate(cls, element):
        eigenvalues_element = element.find('eigenvalues')
        eigenvalues_array = Array.from_element(eigenvalues_element.find('array'))

        array = Array.from_element(element.find('array'))

        return cls(eigenvalues_array, array)


class Calculation(BaseEntry):
    tag = 'calculation'

    def __init__(self, scsteps, structure, forces, stress, energy, time, eigenvalues=None, dos=None, projected=None):
        self.scsteps = scsteps
        self.structure = structure
        self.forces = forces
        self.stress = stress
        self.energy = energy
        self.time = time
        self.eigenvalues = eigenvalues
        self.dos = dos
        self.projected = projected

    @classmethod
    def _instantiate(cls, element):
        scsteps = []
        for scstep_element in element.iter('scstep'):
            scstep = SCStep.from_element(scstep_element)
            scsteps.append(scstep)

        structure = Structure.from_element(element.find('structure'))

        forces_varray = from_element(element.find("varray[@name='forces']"))
        forces = forces_varray.value
        stress_varray = from_element(element.find("varray[@name='stress']"))
        stress = stress_varray.value

        energy = Energy.from_element(element.find('energy'))
        time = Time.from_element(element.find('time'))

        eigenvalues = None
        if element.find('eigenvalues') is not None:
            eigenvalues = Eigenvalues.from_element(element.find('eigenvalues'))

        dos = None
        if element.find('dos') is not None:
            dos = DOS.from_element(element.find('dos'))

        projected = None
        if element.find('projected') is not None:
            projected = Projected.from_element(element.find('projected'))

        return cls(scsteps, structure, forces, stress, energy, time, eigenvalues, dos, projected)


class Modeling(BaseEntry):
    tag = 'modeling'

    def __init__(self, generator, incar, kpoints, parameters, atominfo, structures, calculations):
        self.generator = generator
        self.incar = incar
        self.kpoints = kpoints
        self.parameters = parameters
        self.atominfo = atominfo
        self.structures = structures
        self.calculations = calculations

        self.eigenvalues = None
        self.dos = None
        self.projected = None
        for calculation in calculations:
            if calculation.eigenvalues is not None:
                self.eigenvalues = calculation.eigenvalues
            if calculation.dos is not None:
                self.dos = calculation.dos
            if calculation.projected is not None:
                self.projected = calculation.projected

    @classmethod
    def _instantiate(cls, element):
        generator = ItemSet.from_element(element.find('generator'))
        incar = ItemSet.from_element(element.find('incar'))
        kpoints = Kpoints.from_element(element.find('kpoints'))
        parameters = Parameters.from_element(element.find('parameters'))
        atominfo = Atominfo.from_element(element.find('atominfo'))

        structures = []
        for structure_element in element.iter('structure'):
            structure = Structure.from_element(structure_element)
            structures.append(structure)

        calculations = []
        for calculation_element in element.iter('calculation'):
            calculation = Calculation.from_element(calculation_element)
            calculations.append(calculation)

        return cls(generator, incar, kpoints, parameters, atominfo, structures, calculations)
