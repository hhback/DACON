# 파이프라인 구조 설명

## 만약 모델의 구현을 재현할 경우
- train.sh 실행

## 학습된 모델이 존재한다면 학습없이 Inference만 할 경우
- inference.sh

### 1. Setting Environment
- python version >= 3.6

#### 1-1. Make virtual env
``` 
$ python3 -m venv pyenv
$ source ./pyenv/bin/activate
``` 
#### 1-2. Install requirements
``` 
$ (pyenv) pip install --upgrade pip
$ (pyenv) pip install -r requirements.txt 
``` 
#### 1-3. Train Run Shell

``` 
$ (pyenv) sh ./train.sh
``` 

or

``` 
$ (pyenv) python ./src/preprocess.py
$ (pyenv) python ./src/train.py
``` 

#### 1-4. Inference Run Shell
``` 
$ (pyenv) sh ./inference.sh
``` 

or

``` 
$ (pyenv) python ./src/preprocess.py
$ (pyenv) python ./src/inference.py
``` 

### 2. py file
```
feature.py : feature engineering class py
model.py : model class py
preprocess.py : preprocess activate code
train.py : train activate code
utils.py : utils func py
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
