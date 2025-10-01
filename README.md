# msBayesImpute (Python version)

<img width="2802" height="1598" alt="msBayesImpute architecture" src="https://github.com/user-attachments/assets/1b4f541f-57b9-40c0-9e0e-0ab35029ac44" />

**msBayesImpute** is a versatile framework for handling missing values in **mass spectrometry (MS) proteomics data**.  
It integrates **probabilistic dropout models** with **Bayesian matrix factorization** in a fully data-driven manner,  
allowing it to account for both *missing at random (MAR)* and *missing not at random (MNAR)* patterns.  

This repository contains the **Python implementation** of msBayesImpute, built on [Pyro](https://pyro.ai), a probabilistic programming language.  
The **R version** is available here: [msBayesImpute (R package)](https://github.com/Lu-Group-UKHD/msBayesImpute).  

---

## Repository structure

```bash
msbayesimputepy/
├── data/                       # Example dataset (HeLa cell line proteomics data)
├── msbayesimputepy/            # Python implementation of msBayesImpute
├── msbayesimputepy.egg-info/   # Metadata for the Python package
├── dist/                       # Pre-built Python wheel package
├── vignettes/                  # Example usage (see quick_guide_python.ipynb)
├── requirements.txt            # Package dependencies
└── README.md
```

---

## Installation

Install the Python package from the pre-built wheel in the `dist/` folder:

```bash
pip install dist/msbayesimputepy-0.1.0-py3-none-any.whl
```

---

## Getting started

- See the Jupyter notebook in `vignettes/quick_guide_python.ipynb` for a quick start.  
- Example dataset: provided in the `data/` folder (HeLa cell line proteomics).  

---

## Citation

If you use **msBayesImpute** in your research, please cite:  
*He J, et al. bioRxiv (2025). msBayesImpute: A Versatile Framework for Addressing Missing Values in Biomedical Mass Spectrometry Proteomics Data*

---

