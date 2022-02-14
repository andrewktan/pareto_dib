from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="pareto_dib",
    version='0.0.1',
    author="A. K. Tan",
    author_email='aktan@mit.edu',
    packages=['pareto_dib'],
    scripts=[],
    url="https://github.com/andrewktan",
    license="LICENSE.txt",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=['numpy',
                      'scipy<=1.7',
                      'sortedcontainers',
                      ],
    package_dir={'pareto_dib': 'pareto_dib'},
    test_suite="pareto_dib.test",
)
