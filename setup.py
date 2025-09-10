import setuptools

# Read the contents of the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="plf",
    version="0.1.0", # Or a more appropriate version
    author="stilxam",
    author_email="author@example.com", # Placeholder email
    description="Piecewise-Linear-Functions in JAX",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stilxam/plf",
    project_urls={
        "Bug Tracker": "https://github.com/stilxam/plf/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT from the pyproject.toml
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
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
