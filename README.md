# title_generate
This is an example of using tf.keras to build an encoder-decoder model with attention mechanism to generate the title from the abstract.
If you want to get better performance, I suggest to augment the dataset, even build the pre-trained word embeddings from the specific domain. 

title_parameter.py  reserves the hyper-parameters

title_dataset.py builds data set from title.json

title_train.py trains a model and save it to a file

title-predict.py is an example for using the model to generate title from abstract
