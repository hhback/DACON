# 파이프라인 구조 설명

### 1. Setting Environment
- python version >= 3.6

1. Make virtual env
``` 
$ python3 -m venv pyenv
$ source ./pyenv/bin/activate
``` 
2. Install requirements
``` 
$ (pyenv) pip install --upgrade pip
$ (pyenv) pip install -r requirements.txt 
``` 
3. Train Run Shell
``` 
$ (pyenv) sh ./train.sh
``` 

or

``` 
$ (pyenv) python ./src/preprocess.py
$ (pyenv) python ./src/train.py
``` 

4. Inference Run Shell
``` 
$ (pyenv) sh ./inference.sh
``` 

``` 
$ (pyenv) python ./src/preprocess.py
$ (pyenv) python ./src/inference.py
``` 

### 2. py file
```
feature.py : feature engineering class
model.py : model class
preprocess.py : preprocess activate code
train.py : train activate code
utils.py : utils func
inference.py : inference activate code
```

### 3. requirements.txt
```
numpy
pandas
tqdm
scikit-learn
lightgbm==3.3.2
xgboost==1.6.1
catboost==1.0.6
```
