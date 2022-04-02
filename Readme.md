# Reverse filtering for deblurring 
Python code implementing five reverse filtering (or defiltering) schemes for noisy image deblurring. 

## Dependencies
The code depends on a few common Python libraries (NumPy, ...). Install them via ```pip``` and the provided ```requirements.txt``` file. 
For example 
```bash
$ python -m venv bbdeblur_venv
$ ./bbdeblur_venv/Scripts/activate
$ pip install -r requirements.txt
```

## How to run? 
One possibility is to execute the scripts in the sub-directory ```src``` 
```bash
$ cd src
$ python test_color.py
```

Another possibility is to use the Python notebook, provided in the sub-directory ```notebooks```. The notebook works on Google Colab (with the default installation). 

## Notes 
See also the more complete MATLAB implementation in this [repo](https://github.com/fayolle/bbDeblur). 