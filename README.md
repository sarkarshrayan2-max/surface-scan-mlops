python version - <3.11 , >3.12
App flow 
chmapion model 
I/p image = 224*224*3
Efficient net b0 
hyper params:
1)epoch-10
2)learning rate le10-4
3)include stop flase 
4)frozen_backbone=False
5)GlobalAverage pool
6)Dropout(0.30)
7)Dense(6,Softmax)
Eval metrics=
F1:99.54


raw dataset
     |
create spilts / split generation logic
     |
data_validation.py
     |
preprocessing.py
     |
model_factory.py
     |
train..py
     |
evaluation.py
     |
predict.py
     |
gradcam.py
     |
reports+model artifactd + dimmension_ready interferance     

