# ALPACA Kernel 2 for Jupyter

This package provides the ALPACA kernel for Jupyter.

## Installation via conda


1. `conda install -c twh alpaca_kernel_2`'
2. `conda install ipykernel # Nice to have`
3. `python -m alpaca_kernel install`
4. `python -m ipykernel install`

## Installing environment

1. Get [`requirements_windows_minimal.txt`](https://raw.githubusercontent.com/twhoekstra/alpaca_kernel_2/main/requirements_windows_minimal.txt)
2. `conda config --add channels conda-forge`
4. `conda create -n my_env --file "requirements_windows_minimal.txt" --channel twh`
5. `conda activate my_env`
6. `python -m alpaca_kernel install`
7. `python -m ipykernel install`

## Installation from source (For development)

1. `git clone https://github.com/twhoekstra/alpaca_kernel_2.git`
2. `cd alpaca_kernel_2`
3. `pip install -e .`
4. `python -m alpaca_kernel install`
5. `python -m ipykernel install`

## Demonstration
Check out the functionality of the kernel using [DEMO.ipynb](https://raw.githubusercontent.com/twhoekstra/alpaca_kernel_2/main/DEMO.ipynb)