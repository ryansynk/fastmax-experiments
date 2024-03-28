from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(name='fastmax_cpu',
      ext_modules=[cpp_extension.CppExtension('fastmax_cpu', ['fastmax.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
