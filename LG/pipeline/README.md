# 파이프라인 구조 설명

#### 1. Setting Environment
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
3. Run Shell
``` 
$ (pyenv) sh ./bin/run
``` 


#### 2. py file
```
feature.py : feature engineering class
model.py : model class
preprocess.py : preprocess activate code
train.py : train activate code
utils.py : utils func
```

#### 3. requirements.txt
```
run.sh : run this shell file
```
