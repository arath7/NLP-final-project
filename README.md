# CS4120: Natural Language Processing
## Tishya Kasliwal and Ananya Rath

## About:
- This repository develops takes a 2-pronged approach to developing models to map natural language commands to images of autonomous vehicle scenes and is built on top of the Talk2Car baseline model. 
- In the main branch, within the 'talk2car/baseline' folder are the models corresponding to the research performed on the image encoder model. These models consist of train.py, train_fusion.py, and train_resnet50.py. These models correspondingly are the baseline model, final image encoder model discussed in the research paper, and a model with changes made to the Resnet CNN architecture used in model development. 
- Additionally, when running the train_fusion.py, the dataset_fusion.py dataset file should be used correspondingly. To avoid errors in running models, either rename the run script located in the NLP-final-project-model-runner.ipynb notebook and corresponding train.py file, or rename files to simply train.py and dataset.py when running but rename the existing train.py and dataset.py to other names for avoiding confusion.
- In the Bert branch, this branch contains the code modifications for the text encoder. NLP-final-project/talk2car/baseline/models/ contains the architecture for the SBERT and CLIP text encoders in bert.py and clip.py respectively. These were imported in train.py to test out both encoders.

## Running the Models:
- If running the models locally, follow the Talk2Car instructions here: https://github.com/talk2car/Talk2Car
- If running the models on a Cloud GPU or other virtual instance, use the NLP-final-project-model-runner.ipynb file to mount this folder as a zipped folder to your Google drive and run it from there following the instructions within the notebook
