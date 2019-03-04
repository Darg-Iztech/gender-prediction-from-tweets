# RNN and Captioning for Gender Classification
Gender Classification From Tweets and Posted Images

## Requirements
- Python 2.7 (it may work on python3 as well, not guaranteed)
- (Preferable) CUDA 9.0 and Nvidia GPU compatible with CUDA 
- Word embeddings, we prefer GLoVe (https://nlp.stanford.edu/projects/glove/)
- As a corpus to test, we used PAN2018 Author Profiling dataset (https://pan.webis.de/clef18/pan18-web/author-profiling.html)

#### _Packages_:
* Tensorflow 1.5 or higher
* Gensim
* Numpy
* NLTK
* Matplotlib

## Usage
- To change parameters you can use parameters.py file, it has all paths, hyperparameters, dimension fields.

- Run.sh make it easy to test several languages and parameters in one big run. After each language Training&Testing it calls model deleter which deletes each saved model but the first 5.

- Making ```Optimize=True``` means that you are going to test some hyperparameters and save models that got higher than your ```model_save_threshold```. If you make it false you will just train a model without saving any weigths.

- To switch into other models that have been tested for Gender Prediction problem, you can switch branches, we hope their names are self-exploratory.
