from setuptools import setup, find_packages

setup(
    name = "msbayesimputepy",
    version = "0.1.0",
    packages = find_packages(),
    install_requires = ["pandas", "numpy==1.26.2", "torch", "pyro-ppl>=1.9.0", "scikit-learn", "scipy", "seaborn", "matplotlib"],
    author = "Jiaojiao He",
    author_email = "jiaojiao.he918@gmail.com",
    description = "msBayesImpute: a versatile framework for addressing missing values in biomedical mass-spectrometry proteomic data.",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    # url = "https://github.com/yourusername/my_package",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)