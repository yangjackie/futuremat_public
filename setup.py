from setuptools import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Distutils import build_ext
    import Cython.Compiler.Options

    Cython.Compiler.Options.annotate = True
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("core.models.vector3d", ["core/models/vector3d.pyx"], language='c++',include_dirs=[numpy.get_include()]),
        Extension("core.models.matrix3d", ["core/models/matrix3d.pyx"], language='c++'),
        Extension("core.models.lattice", ["core/models/lattice.pyx"], language='c++'),
        #Extension("core.internal.molecule.analysis",
        #          ["core/internal/molecule/analysis.pyx"],language='c++',include_dirs=[numpy.get_include()])
    ]
    cmdclass.update({'build_ext': build_ext})

setup(
    name='futuremat',
    version='1.0',
    packages=['futuremat', 'futuremat.core', 'furturemat.twodPV'],
    entry_points={
        'console_scripts': [
            'futuremat = futuremat.__main__:main'
        ]
    },
    url='',
    license='',
    author='jackyang',
    author_email='jianliang.yang1@unsw.edu.au',
    description='',
    cmdclass=cmdclass,
    ext_modules=ext_modules,

)
