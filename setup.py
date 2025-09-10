import setuptools

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="plf",
    version="0.0.1",
    author="Maxwell Litsios",
    author_email="m.l.h.litsios@student.tue.nl",
    description="Gradient optimized piecewise linear fitting in JAX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stilxam/plf",
    project_urls={
        "Bug Tracker": "https://github.com/stilxam/plf/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=[
        "jax",
        "equinox",
        "optax",
        "jaxtyping",
    ],
)
