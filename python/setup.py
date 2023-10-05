import os
import sys
import sysconfig
from pathlib import Path

from setuptools import find_packages

# need to use distutils.core for correct placement of cython dll
if "--inplace" in sys.argv:
    from distutils.core import setup
    from distutils.extension import Extension
else:
    from setuptools import setup
    from setuptools.extension import Extension


def config_cython():
    user_home_path = str(Path.home())
    sys_cflags = sysconfig.get_config_var("CFLAGS")
    try:
        from Cython.Build import cythonize

        ret = []
        path = "quartz/_cython"
        for fn in os.listdir(path):
            if not fn.endswith(".pyx"):
                continue
            ret.append(
                Extension(
                    "quartz.%s" % fn[:-4],
                    ["%s/%s" % (path, fn)],
                    include_dirs=[
                        "../src/quartz/",
                        "/usr/local/include/",
                        os.path.join(user_home_path, "usr_local/include"),
                    ],
                    libraries=["quartz_runtime"],
                    library_dirs=[
                        "/usr/local/lib/",
                        os.path.join(user_home_path, "usr_local/lib"),
                    ],
                    extra_compile_args=["-std=c++17"],
                    extra_link_args=[],
                    language="c++",
                )
            )
        return cythonize(
            ret,
            compiler_directives={"language_level": 3},
            emit_linenums=True,
        )
    except ImportError:
        print("WARNING: cython is not installed!!!")
        return []


setup_args = {}

setup(
    name='quartz',
    version="0.1.0",
    description="Quartz: Superoptimization of Quantum Circuits",
    zip_safe=False,
    install_requires=[],
    packages=['quartz'],  # find_packages()
    url='https://github.com/quantum-compiler/quartz',
    ext_modules=config_cython(),
)
