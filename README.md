# End_to_End_ML_MLOPS_Project

#workflows
1. Update config.yaml
2. update schema.yaml
3. update params.yaml
4. update entity
5. update configuration manager in src config
6. update the components
7. update pipeline
8. update main.py
9. update app.py

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/Poojitha12345678/End_to_End_ML_MLOPS_Project
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mlproj python=3.8 -y
```

```bash
conda activate mlproj
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```



## MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)


##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/Poojitha12345678/End_to_End_ML_MLOPS_Project.mlflow \
MLFLOW_TRACKING_USERNAME=Poojitha12345678 \
MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/Poojitha12345678/End_to_End_ML_MLOPS_Project.mlflow \

export MLFLOW_TRACKING_USERNAME=Poojitha12345678 

export MLFLOW_TRACKING_PASSWORD=ef701f6a002ce731bd869c44a6ad166b3a5a05ed

```



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess