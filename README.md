# One-shot Logo Detection
Academic project for "Machine Learning" and "Sistemi Intelligenti per Internet" (Intelligent Internet Systems) courses, Roma Tre University.

On a high level, the project is a Pytorch implementation of the 2019 https://arxiv.org/pdf/1811.01395.pdf paper, by Bhunia et al.

The model is configured to be trained using the "FlickrLogos-32-v2" dataset, which you can obtain following the instructions on the following link: https://www.uni-augsburg.de/en/fakultaet/fai/informatik/prof/mmc/research/datensatze/flickrlogos/. You can obviously configure your own, by tweaking @/utils/dataset_loader.py and @/config/config.yaml appropriately. The same can be done for the testing datasets.
Then, the model can be trained by running the @/train.ipynb Jupyter Notebook.

To test the model you can run the test.ipynb Jupyter Notebook.

To visualize model predictions you can run predict.ipynb Jupyter Notebook.
