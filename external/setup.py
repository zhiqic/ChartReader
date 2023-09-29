import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "bbox", 
        ["bbox.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]
    ),
    Extension(
        "nms", 
        ["nms.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"]
    )
]

setup(
    name="coco",
    ext_modules=cythonize(extensions, language_level="3"),
    include_dirs=[numpy.get_include()]
)
