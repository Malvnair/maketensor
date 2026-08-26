# maketensor

`maketensor` prepares astronomical difference-image data for training machine-learning models to detect faint Trans-Neptunian Objects (TNOs).

The pipeline converts CLASSY FITS observations into reusable HDF5 cutout sequences, optionally copies those files to scratch storage, injects synthetic moving TNOs into the real observations, and loads the resulting data as PyTorch tensors.

The repository contains four main scripts:

- `fits2hdf5.py` — converts FITS observations into raw HDF5 cutout shards.
- `stage2scratch.py` — optionally copies the HDF5 shards from ARC storage to scratch.
- `tno_injection.py` — generates and injects synthetic moving TNOs into the real image sequences.
- `pytorch_hdf5_loader.py` — loads the HDF5 data, performs optional online injection, and returns PyTorch tensors for training.

## Documentation

Detailed documentation for the pipeline and structure is available here:

((https://docs.google.com/document/d/1LI_N1iozdtkE536LauWMsUBAXZZzsr1WuyoTbBc8sls/edit?usp=sharing))

## TRIPPy

This repository uses TRIPPy, developed by Fraser et al. (2016), "TRIPPy: Trailed Image Photometry in Python."

https://github.com/fraserw/trippy

TRIPPy is used to restore the point-spread function (PSF) for each observation and generate the trailed source models used when planting synthetic TNOs. This allows the injected objects to reproduce the shape and motion-blurring expected in the real observations.


## Dependencies

The main Python dependencies are:

- `numpy` — array operations, random sampling, and numerical calculations.
- `astropy` — reading FITS files and handling WCS coordinate transformations.
- `h5py` — creating and reading HDF5 shard files.
- `torch` — PyTorch Dataset and DataLoader support for model training.
- `matplotlib` — generating diagnostic and example images.
- `TRIPPy` — PSF restoration and synthetic trailed-source generation.

The code was developed for the NRC/ARC computing environment. 
Data paths in the scripts are specific to ARC and will need to be changed if the repository is run elsewhere.
Update the source, output, and TRIPPy paths to match your own username and directory locations.


## Pipeline

The basic workflow is:

FITS data → HDF5 shards → optional scratch staging → synthetic TNO injection → PyTorch tensors
