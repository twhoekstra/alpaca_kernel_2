# ALPACA Kernel 2 for Jupyter

This package provides the ALPACA kernel for Jupyter.

## Installation via conda


1. `conda install -c twh alpaca_kernel_2`'
2. `conda install ipykernel # Nice to have`
3. `python -m alpaca install`
4. `python -m ipykernel install`

## Installing environment

1. Get [`requirements_windows.txt`](https://raw.githubusercontent.com/twhoekstra/alpaca_kernel_2/main/requirements_windows.txt)
2. `conda create -n my_env --file "requirements_windows.txt" --channel twh`
3. `conda activate my_env`
4. `python -m alpaca install`
4. `python -m ipykernel install`

## Installation from source (For development)

1. `git clone https://github.com/twhoekstra/alpaca_kernel_2.git`
2. `cd alpaca_kernel_2`
3. `pip install -e .`
4. `python -m alpaca install`
5. `python -m ipykernel install`

## Demonstration
Check out the functionality of the kernel using [DEMO.ipynb](https://raw.githubusercontent.com/twhoekstra/alpaca_kernel_2/main/DEMO.ipynb)