# Source code required to reproduce the figures and results in the 2020 Artificial Life conference submission "Host-Pathongen Co-evolution Inspired Algorithm Enables Robust GAN Training"

The installation requires a scientific Python 3 installation, with PyTorch installation and CUDA
 cores available, as well as a MogoDB container running. The MogoDB container can be configured
  and started with the `docker-compose up` command from within the source of the file.
  
Following variables need to be adjusted for the environment in which the code is executed: 
 - Environment variable `MONGOROOTPASS`
 - `trace_dump_file` in `src.results_analysis.py` and in `src.evolutionary_arena.py`
 - `backflow_log` in `src.results_analysis.py`
 - `image_folder` in `src.evolutionary_arena.py` and `src.train_and_match.py`
 - `image_samples_folder` in `src.train_and_match.py`
 - `device` parameter on line 844 in `src.evolutionary_arena.py`
 
 The code execution is as follows: 
 - `python -m src.evolutionary_arena`
 - upon completion `python -m src.results_analysis`