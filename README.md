# vasputils

Pre-and-Post processing tools for Vienna Ab initio Simulation Package (VASP)

Table of Contents
-----------------

  * [Requirements](#requirements)
  * [Usage](#usage)
  * [License](#license)


Requirements
------------

vasputils requires the following to run:

  * python 2.7
  * numpy

Installation
-----

```sh
pip install git+git://github.com/akiraakaishi/vasputils.git
```

Usage
-----

```sh
chg_split ./CHG
vasprun dos
vasprun --file somewhere/vasprun.xml eigenval --energy=-3.0:3.0
```



License
-------

Paddington is licensed under the [MIT](#) license.  
Copyright &copy; 2016, Akira Akaishi
