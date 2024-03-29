from setuptools import setup, find_packages

setup(
    name="phmg_thesis",
    version="0.1",
    packages=find_packages(),
    description="Monolithic multigrid preconditioners for high-order Stokes systems.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Alexey Voronin",
    author_email="voronin2@illinois.edu",
    url="https://github.com/alexey-voronin/phmg_thesis",
    install_requires=[
        "firedrake>=0.13.0",
        "pandas>=1.4.0",
    ],
    package_data={
        "sysmg": ["data/**/*.sh", "data/**/*.py", "data/**/*.json", "data/**/*/.md"],
    },
)
