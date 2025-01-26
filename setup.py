import platform
import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import shutil

install_requires = [
    'matplotlib==3.5.3',
    'networkx==3.1',
    'numpy==1.25.2',
    'openpyxl==3.1.3',
    'pandas==2.2.0',
    'PyYAML==6.0.1',
    'scikit_learn>=1.5.0',
    'scipy==1.12.0',
    'statsmodels==0.14.0',
    'wget==3.2'
]

is_macos = platform.system() == "Darwin"
cpp_extension = Extension(
    "cAdmmUpdate",
    sources=["./src/cAdmmUpdate.cpp"],
    language="c++",
    extra_compile_args=["-std=c++11", "-fPIC", "-w"],
    extra_link_args=["-bundle"] if is_macos else ["-shared"]
)

class CustomBuildExt(_build_ext):
    def get_ext_filename(self, ext_name):
        if os.environ.get("IS_DOCKER_EXT", "false") == "true":
            return os.path.join("app", "src", f"{ext_name}.so")
        else:
            return os.path.join("..", "..", "src", f"{ext_name}.so")

setup(
    name="high_dim_svar",
    version="0.1.0",
    author="Jiahe Lin",
    author_email="jiahelin@umich.edu",
    description="An ADMM-based algorithm that performs structural discovery of DAGs, given partial ordering",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.10',
    zip_safe=False,
    ext_modules=[cpp_extension],
    cmdclass={"build_ext": CustomBuildExt},
)

for folder_name in ["./build", "./high_dim_svar.egg-info"]:
    try:
        shutil.rmtree(folder_name)
    except Exception as e:
        print(f'The following error occured: {str(e)} but okay to ignore')

try:
    os.remove("cAdmmUpdate.so")
except Exception as e:
    print(f'The following error occured: {str(e)} but okay to ignore')
