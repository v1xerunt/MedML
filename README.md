# MedML
Source code, knowledge graphs and feature lists for the paper MedML: Fusing Medical Knowledge and Machine Learning Models for Early Pediatric COVID-19 Hospitalization and Severity Prediction

## Requirements

* Install python, pytorch. We use Python 3.6.15, Pytorch 1.7.1.
* If you plan to use GPU computation, install CUDA

## Data preparation

- ```x_train```/```x_val```/```x_test``` are feature matrices to train the GNN model. Each matrix should has shape ```[N, F_g]```, where ```N``` denotes the number of train/val/test patients, and ```F_g``` denotes the feature dimension.
- ```demo_train```/```demo_val```/```demo_test``` are additional feature matrices. These matrices stores the demographical data, which are not part of the knowledge graph but still will be used to train the model. Each matrix should has shape ```[N, F_d]```, where ```N``` denotes the number of train/val/test patients, and ```F_d``` denotes the feature dimension.
- ```y_train```/```y_val```/```y_test``` are labels vectors. Each vector's length is ```N```, where ```N``` denotes the number of train/val/test patients.
- ```edge_pair.pkl``` stores the knowledge graph structure. It is a list of tuples. Each tuple ```(i,j)``` stores the edge from node ```i``` to ```j``` in the knowledge graph. Node index ```i``` and ```j``` should be corresponded to the feature index in the ```x``` matrices.

The data should be stored in the ```data``` folder. We provide a few dummy data for reference.

## Training MedML

The training functions and hyperparameters are in the ```train_medml.py```. 

## License

Please refer to License.txt or License.docx for research usage.

