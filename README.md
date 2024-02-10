# high-dim-structural-VAR-partial-ordering

Code repository for paper titled **"Structural Discovery with Partial Ordering Information for Time-Dependent Data with Convergence Guarantees"**, authored by Jiahe Lin, Huitian Lei and George Michailidis. https://arxiv.org/abs/2311.15434. *Journal of Computational and Graphical Statistics, 2024*
```
@article{lin2024Structural,
  title={Structural discovery with partial ordering information for time-dependent data with convergence guarantees},
  author = {Jiahe Lin, Huitian Lei and George Michailidis},
  title = {Structural Discovery with Partial Ordering Information for Time-Dependent Data with Convergence Guarantees},
  journal = {Journal of Computational and Graphical Statistics},
  volume = {0},
  number = {0},
  pages = {1-10},
  year = {2024},
  publisher = {Taylor & Francis},
  doi = {10.1080/10618600.2023.2301097}
}
```

## Setup

* Step 1: Configure the python environment (assuming one of anaconda/miniconda/miniforge has been installed):
  ```console
  conda create -n svar python=3.10
  conda activate svar
  conda install pyyaml numpy pandas statsmodels scikit-learn networkx matplotlib openpyxl
  pip install wget
  ```
  See also `requirements.txt`.

* Step 2: Compile the source file; this step creates the necessary shared library based on the source cpp 
  ```
  bash setup.sh
  ```

## Outline of the Repo
To facilitate the users in traversing the repository, we provide a brief outline for the organization of this repository
* `src/`: hosts the core implementation of the proposed methodology 
* `utils/`: hosts the utility functions/classes related to graphs, synthetic data generation and performance evaluation
* `configs/`: hosts the config file for synthetic data generation and sample configs for performing model fitting
* `data/`: hosts the scripts for pre-processing the datasets used in the real data experiments. This is also the location where the raw data should be stored (see section Real Data Experiments for more details)


## Synthetic Data Generation
The following command allows one to simulate data according to the description in the synthetic data experiment section. 
```console
python -u simulate_data.py --datasets=ds3
```
* To simulate multiple datasets at once, specify them through a comma separated string; e.g., `--datasets=ds1,ds2,ds3`
* The default config file being used is `configs/datasets.yaml`; each section key corresponds to the specific setting of interest. By default, 1 replicate of the designated dataset(s) will be generated and saved in their respective folders under `data/sim/${DATASET_OF_INTEREST}`
* For synthetic data experiment, the partial ordering information is saved in `data/sim/${DATASE_OF_INTEREST}/graph_info.pickle`
* Pass-in any alternative configuration file through `--config_override`

## Model Fitting
The following command allows one to perform estimation on a specific synthetic dataset:
```console
python -u run_sim.py --ds_str=ds3 --train_size=200 ## without standardization
python -u run_sim.py --ds_str=ds3 --train_size=200 --standardize ## with standardization
```
* run parameters are specified through `configs/${DATASE_OF_INTEREST}.yaml`. 
* section `default` specifies the default parameters used in the ADMM step, which are typically fairly robust.
* setting-specific parameters are specified under their respective sections, in the format of `${DATASE_OF_INTEREST}-${TRAIN_SIZE}${STANDARDIZATION_SUFFIX}`. 
* One can selectively overrides the default parameters, by providing the values to the corresponding keys.  
* Add flag `--report` to get the performance evaluation this replicate. 

Notes: see L65 for how the model class is instantiated; see L97 for triggering model fitting through the `.fit()` method, with which the structural and the lag components are being estimated. 

## Real Data

Both real datasets used in this paper are publicly available. 

### US Macroeconomic Dataset
The dataset is available https://research.stlouisfed.org/econ/mccracken/fred-databases/; download a copy of the vintage of interest and save it as `data/macro/YYYYMM_Qraw.csv`

To prepare data for the experiment where the US Macroeconomic Dataset is used, use the following command; specify the data vintage YYYYMM through `--vintage`. 
```console
python -u data/macro/prep_macro_data.py --vintage=202209
```
* In the case the raw data is not saved, data will be automatically downloaded and saved as `data/macro/CURRENT_Qraw.csv`, where CURRENT corresponds to the **current month** in YYYYMM format; as such, the specified vintage will be overriden. Note that the download step relies on `wget`. 

### DREAM4 Dataset
The dataset can be accessed through R-biocManager. 

The following command allows one to install the BiocManager and necessary libraries
```console
Rscript --vanilla data/dream4/setup.R
```

To extract the data, refer to script `data/dream4/data_extract.R`; note that one needs to make necessary changes to L10-L14 (see also below) and execute the script in the R console to extract the five datasets sequentially
```R
## 'dream4_100_01', 'dream4_100_02', 'dream4_100_03', 'dream4_100_04', 'dream4_100_05'
filename = 'dream4_100_05'
data(dream4_100_05)
mtx.all = assays(dream4_100_05)[[1]]
mtx.goldStandard = metadata(dream4_100_05)[[1]]
```
Each dataset will be saved into a designated Excel file under `data/dream4/.`

To prep the data, execute the following command, which will save down the corresponding pickle file for subsequent model fitting. 
```console
python -u data/dream4/prep_dream4_data.py
```

## Contact
* For general questions on the paper, contact George Michailidis [gmichail AT ucla DOT edu] 
* For questions on the code implementation, contact Jiahe Lin [jiahelin AT umich DOT edu]