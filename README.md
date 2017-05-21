# RBM
My implementation for RBM - Restricted Bolzmann Machine for Collaborative Filtering
- RBM is describe in the paper http://www.machinelearning.org/proceedings/icml2007/papers/407.pdf 
- On how to train RBM, you can find it at http://image.diku.dk/igel/paper/TRBMAI.pdf
- The data for this repo is from CSEP 546: Data Mining, detail is in http://courses.cs.washington.edu/courses/csep546/10sp/hw1/

# Dependency 
The implementation run in Tensorflow and Numpy. If you are using windows, please install Anaconda and then Tensorflow and Numpy. If you are on linux machine, just using pip to install but still Anaconda is recommended. Detail can be found below.
- Anaconda https://www.continuum.io/downloads
- Tensorflow https://www.tensorflow.org/install/
- Numpy http://www.numpy.org/

# How to run
To run the program just use "python main.py" in the directory. All the data have been saved in data folder, you can read the description of the data on the description.txt and README file on the data/raw_data folder. 

# Credit
Special thank to Blackecho and his gist "Restricted Boltzmann Machine implementation in TensorFlow, before and after code refactoring." you can find it here: https://gist.github.com/blackecho/db85fab069bd2d6fb3e7
