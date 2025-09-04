# Fine-tuning Florence-2 for VQA (Visual Question Answering) using the Azure ML Python SDK and MLflow

Released by Microsoft in mid-June 2024 under an MIT license, Florence-2 is less than 1B in size (0.23B for the base model and 0.77B for the large model) and is efficient for vision and vision-language tasks (OCR, captioning, object detection, instance segmentation, and so on).

All of Florence-2's weights are publicly available, so you can fine-tune it quickly and easily. However, many people struggle with fine-tuning the latest SLM/multi-modal models, including Florence-2, in Azure ML studio. So, we want to walk through a step-by-step guide on how to quickly and easily train and serve from end-to-end in Azure ML.

## 1. Training preparation <a href="#community-4181123-toc-hid-1244552293" id="community-4181123-toc-hid-1244552293"></a>

***

### 1.1. Preliminaries: Azure ML Python SDK v2 <a href="#community-4181123-toc-hid-1944025118" id="community-4181123-toc-hid-1944025118"></a>

Azure ML Python SDK v2 is easy to use once you get the hang of it. When an `MLClient` instance is created to manipulate AzureML, the operation corresponding to the asset is executed asynchronously through the `create_or_update` function. Please see code snippets below.

```python
ml_client = MLClient.from_config(credential)

### 1. Training phase
# Create an environment asset
ml_client.environments.create_or_update(env_docker_image)

# Create a data asset
ml_client.data.create_or_update(data)

# Create a compute cluster
ml_client.compute.begin_create_or_update(compute)

# Start a training job
ml_client.jobs.create_or_update(job)

### 2. Serving phase
# Create a model asset
ml_client.models.create_or_update(run_model)

# Create an endpoint
ml_client.begin_create_or_update(endpoint)

# Create a deployment
ml_client.online_endpoints.begin_create_or_update(endpoint)  
```

### 1.2. Data asset <a href="#community-4181123-toc-hid-136570655" id="community-4181123-toc-hid-136570655"></a>

Model training/validation datasets can be uploaded directly locally, or registered as your Azure ML workspace Data asset. Data asset enables versioning of your data, allowing you to track changes to your dataset and revert to previous versions when necessary. This maintains data quality and ensures reproducibility of data analysis.

Data assets are created by referencing data files or directories stored in Datastore. Datastore represents a location that stores external data and can be connected to various Azure data storage services such as Azure Blob Storage, Azure File Share, Azure Data Lake Storage, and OneLake. When you create an Azure ML workspace, four datastores (`workspaceworkingdirectory`, `workspaceartifactstore`, `workspacefilestore`, `workspaceblobstore`) are automatically created by default. Among these, `workspaceblobstore` is Azure Blob Storage, which is used by default when storing model training data or large files.

### 1.3. Environment asset <a href="#community-4181123-toc-hid-126067551" id="community-4181123-toc-hid-126067551"></a>

Azure ML defines Environment Asset in which your code will run. We can use the built-in environment or build a custom environment using Conda specification or Docker image. The pros and cons of Conda and Docker are as follows.

**Conda environment**

* Advantages
  * Simple environment setup: The Conda environment file (`conda.yml`) is mainly used to specify Python packages and Conda packages. The file format is simple and easy to understand, and is suitable for specifying package and version information.
  * Quick setup: The Conda environment automatically manages dependencies and resolves conflicts, so setup is relatively quick and easy.
  * Lightweight environment: Conda environments can be lighter than Docker images because they only install specific packages.
* Disadvantages
  * Limited flexibility: Because the Conda environment focuses on Python packages and Conda packages, it is difficult to handle more complex system-level dependencies.
  * Portability limitations: The Conda environment consists primarily of Python and Conda packages, making it difficult to include other languages or more complex system components.

**Docker environment**

* Advantages
  * High flexibility: Docker allows you to define a complete environment, including all necessary packages and tools, starting at the operating system level. May contain system dependencies, custom settings, non-Python packages, etc.
  * Portability: Docker images run the same everywhere, ensuring environment consistency. This significantly improves reproducibility and portability.
  * Complex environment setup: With Docker, you can set up an environment containing complex applications or multiple services.
* Disadvantages
  * Complex setup: Building and managing Docker images can be more complex than setting up a Conda environment. You need to write a `Dockerfile` and include all required dependencies.
  * Build time: Building a Docker image for the first time can take a long time, especially if the dependency installation process is complex.

In Azure ML, it is important to choose the appropriate method based on the requirements of your project. For simple Python projects, the Conda environment may be sufficient, but if you need complex system dependencies, the Docker environment may be more appropriate. The easiest and fastest way to create a custom Docker image is to make minor modifications to the curated environment. Below is an example.

Please select `acft-hf-nlp-gpu` in the cured environment tab. (Of course, you can choose a different environment.)

![environment\_curated1.png](https://techcommunity.microsoft.com/t5/s/gxcuf89792/images/bS00MTgxMTIzLTU5Njc1MWlFRjAwQTkzNzdEMUI1NzlB?image-dimensions=488x519\&revision=6)

Please copy the `Dockerfile` and `requirements.txt` and modify them as needed.

![environment\_curated2.png](https://techcommunity.microsoft.com/t5/s/gxcuf89792/images/bS00MTgxMTIzLTU5Njc1Mmk5NzY2RkU0RjQ1M0U2NUUz?image-dimensions=999x509\&revision=6)

The code snippet below is the result of modifying the `Dockerfile`.

```applescript
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch222:biweekly.202406.2

USER root

RUN apt-get update && apt-get -y upgrade
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN python -m nltk.downloader punkt
RUN MAX_JOBS=4 pip install flash-attn==2.5.9.post1 --no-build-isolation
```

## 2. Training <a href="#community-4181123-toc-hid-795622817" id="community-4181123-toc-hid-795622817"></a>

***

### 2.1. Training Script with MLflow <a href="#community-4181123-toc-hid-310767068" id="community-4181123-toc-hid-310767068"></a>

Some people may think that they need to make significant changes to their existing training scripts or that the Mlflow toolkit is mandatory, but this is not true. If you are comfortable with your existing training environment, you don't need to adopt Mlflow. Nevertheless, Mlflow is a toolkit that makes training and deploying models on Azure ML very convenient, so we are going to briefly explain it in this post.

\
In the your training script, Use `mlflow.start_run()` to start an experiment in MLflow, and `mlflow.end_run()` to end the experiment when it is finished. Wrapping it in `with` syntax eliminates the need to explicitly call end\_run(). You can perform mlflow logging inside an mlflow block, our training script uses `mlflow.log_params()`, `mlflow.log_metric()`, and `mlflow.log_image()`. For more information, please see [here](https://learn.microsoft.com/azure/machine-learning/how-to-log-view-metrics).

```applescript
import mlflow
...
with mlflow.start_run() as run:
  mlflow.log_params({
    "epochs": epochs,
    "train_batch_size": args.train_batch_size,
    "eval_batch_size": args.eval_batch_size,
    "seed": args.seed,
    "lr_scheduler_type": args.lr_scheduler_type,        
    "grad_accum_steps": grad_accum_steps, 
    "num_training_steps": num_training_steps,
    "num_warmup_steps": num_warmup_steps,
  })    
    
  # Your training code
  for epoch in range(epochs):     
    train_loss = 0.0
    optimizer.zero_grad()

    for step, (inputs, answers) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")):    
      ...
      mlflow.log_metric("train_loss", train_loss)
      mlflow.log_metric("learning_rate", learning_rate)
      mlflow.log_metric("progress", progress)
      ...
      if (step + 1) % save_steps == 0:
        # Log image
        idx = random.randrange(len(val_dataset))
        val_img = val_dataset[idx][-1]
        result = run_example("DocVQA", 'What do you see in this image?', val_dataset[idx][-1])
        val_img_result = create_image_with_text(val_img, json.dumps(result))
        mlflow.log_image(val_img_result, key="DocVQA", step=step)    
```

> **\[Caution]** Florence-2 is a recently released model and does not support mlflow.transformers.log\_model() as of July 2, 2024, when this article is being written! Therefore, you must save the model with the traditional `save_pretrained()`.
>
> Currently, when saving with `save_pretrained()`, additional dependency codes required for model inference are not saved together. So, you need to force it to be saved. See below for a code snippet reflecting these two caveats.

```applescript
model.save_pretrained(model_dir)
processor.save_pretrained(model_dir)

## Should include configuration_florence2.py, modeling_florence2.py, and processing_florence2.py
dependencies_dir = "dependencies"
shutil.copytree(dependencies_dir, model_dir, dirs_exist_ok=True)
```

### 2.2. Create a Compute Cluster and Training Job <a href="#community-4181123-toc-hid-2118221531" id="community-4181123-toc-hid-2118221531"></a>

Once you have finished writing and debugging the training script, you can create a training job. As a baseline, you can use `Standard_NC24ads_A100_v4` with one NVIDIA A100 GPU. Provisioning a LowPriority VM costs just $0.74 per hour in the US East region in July 2024.

The `command()` function is one of the Azure ML main functions used to define and run training tasks. This function specifies the training script and its required environment settings, and allows the job to be run on Azure ML's compute resources.

```applescript
from azure.ai.ml import command
from azure.ai.ml import Input
from azure.ai.ml.entities import ResourceConfiguration

job = command(
    inputs=dict(
        #train_dir=Input(type="uri_folder", path=DATA_DIR), # Get data from local path
        train_dir=Input(path=f"{AZURE_DATA_NAME}@latest"),  # Get data from Data asset
        epoch=d['train']['epoch'],
        train_batch_size=d['train']['train_batch_size'],
        eval_batch_size=d['train']['eval_batch_size'],  
        model_dir=d['train']['model_dir']
    ),
    code="./src_train",  # local path where the code is stored
    compute=azure_compute_cluster_name,
    command="python train_mlflow.py --train_dir ${{inputs.train_dir}} --epochs ${{inputs.epoch}} --train_batch_size ${{inputs.train_batch_size}} --eval_batch_size ${{inputs.eval_batch_size}} --model_dir ${{inputs.model_dir}}",
    #environment="azureml://registries/azureml/environments/acft-hf-nlp-gpu/versions/61", # Use built-in Environment asset
    environment=f"{azure_env_name}@latest",
    distribution={
        "type": "PyTorch",
        "process_count_per_instance": 1, # For multi-gpu training set this to an integer value more than 1
    },
)
returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
```

### 2.3. Check your Training job <a href="#community-4181123-toc-hid-369291302" id="community-4181123-toc-hid-369291302"></a>

Check whether model training is progressing normally through Jobs Asset.

1.  **Overview** tab allows you to view your overall training history. Params are parameters registered in `mlflow.log_params()` in our training script.

    ![train\_job\_overview.png](https://techcommunity.microsoft.com/t5/s/gxcuf89792/images/bS00MTgxMTIzLTU5Njc1NGk5RDEzMzcyRDNCRkZCODYx?image-dimensions=999x412\&revision=6)

    &#x20;
2.  **Metrics** tab allows you to view the metrics registered with `mlflow.log_metric()` at a glance.

    ![train\_job\_metrics.png](https://techcommunity.microsoft.com/t5/s/gxcuf89792/images/bS00MTgxMTIzLTU5Njc1NmkxM0JEMEMwQjE1RUE1Qjkw?image-dimensions=999x604\&revision=6)
3.  **Images** tab allows you to view images saved with `mlflow.log_image()`. We recommend that you save the inference results as an image to check whether the model training is progressing well.

    ![train\_job\_images.png](https://techcommunity.microsoft.com/t5/s/gxcuf89792/images/bS00MTgxMTIzLTU5Njc1OGlCMTEzQzdCM0E3MkJDNEMw?image-dimensions=999x697\&revision=6)

     
4.  **Outputs + logs** tab checks and monitors your model training infrastructure, containers, and code for issues.

    `system_logs` folder records all key activities and events related to the Training cluster, data assets, hosted tools, etc.

    `user_logs` folder mainly plays an important role in storing logs and other files created by users within the training script, increasing transparency of the training process and facilitating debugging and monitoring. This allows users to see a detailed record of the training process and identify and resolve issues when necessary.

    ![traing\_job\_logs.png](https://techcommunity.microsoft.com/t5/s/gxcuf89792/images/bS00MTgxMTIzLTU5Njc1OWkxNkU4Q0FDNkI3RUEwNEVG?image-dimensions=999x384\&revision=6)

\
3\. Serving <a href="#community-4181123-toc-hid-1903475218" id="community-4181123-toc-hid-1903475218"></a>
----------------------------------------------------------------------------------------------------------

***

Once the model training is complete, let's deploy it to the hosting server. If you saved it with MLflow `log_model()`, you can deploy it directly with Mlflow, but in the current transformer and mlflow version, we used the traditional way of saving the model, so we need to deploy it with the custom option.

### 3.1. Inference script <a href="#community-4181123-toc-hid-354186193" id="community-4181123-toc-hid-354186193"></a>

You only need to define two functions, `init()` and `run()`, and write them freely. Although you cannot pass arguments to the `init()` function directly, you can pass the necessary information during initialization through environment variables or configuration files.

```applescript
import os
import re
import json
import torch
import base64
import logging

from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, BitsAndBytesConfig, get_scheduler
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image, ImageDraw, ImageFont

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_example_base64(task_prompt, text_input, base64_image, params):

    max_new_tokens = params["max_new_tokens"]
    num_beams = params["num_beams"]
    
    image = Image.open(BytesIO(base64.b64decode(base64_image)))
    prompt = task_prompt + text_input

    # Ensure the image is in RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        num_beams=num_beams
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    global processor
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # Please provide your model's folder name if there is one
    model_name_or_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "outputs"
    )
    
    model_kwargs = dict(
        trust_remote_code=True,
        revision="refs/pr/6",        
        device_map=device
    )
    
    processor_kwargs = dict(
        trust_remote_code=True,
        revision="refs/pr/6"
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)    

    logging.info("Loaded model.")

def run(json_data: str):
    logging.info("Request received")
    data = json.loads(json_data)
    task_prompt = data["task_prompt"]
    text_input = data["text_input"]
    base64_image = data["image_input"]
    params = data['params']

    generated_text = run_example_base64(task_prompt, text_input, base64_image, params)
    json_result = {"result": str(generated_text)}
    
    return json_result    
```

### 3.2. Register Model <a href="#community-4181123-toc-hid-1453268270" id="community-4181123-toc-hid-1453268270"></a>

Register with the `Model` class of `azure.ai.ml.entities`. Enter the model's path and name when registering and use with `ml_client.models.create_or_update().`

```applescript
def get_or_create_model_asset(ml_client, model_name, job_name, model_dir="outputs", model_type="custom_model", update=False):
    
    try:
        latest_model_version = max([int(m.version) for m in ml_client.models.list(name=model_name)])
        if update:
            raise ResourceExistsError('Found Model asset, but will update the Model.')
        else:
            model_asset = ml_client.models.get(name=model_name, version=latest_model_version)
            print(f"Found Model asset: {model_name}. Will not create again")
    except (ResourceNotFoundError, ResourceExistsError) as e:
        print(f"Exception: {e}")        
        model_path = f"azureml://jobs/{job_name}/outputs/artifacts/paths/{model_dir}/"    
        run_model = Model(
            name=model_name,        
            path=model_path,
            description="Model created from run.",
            type=model_type # mlflow_model, custom_model, triton_model
        )
        model_asset = ml_client.models.create_or_update(run_model)
        print(f"Created Model asset: {model_name}")

    return model_asset
```

### 3.3. Environment asset <a href="#community-4181123-toc-hid-1034244563" id="community-4181123-toc-hid-1034244563"></a>

This is the same as the Environment asset introduced in the previous section. However, model serving requires additional settings for web hosting, so please refer to the code snippet below.

```applescript
FROM mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu118-py38-torch222:biweekly.202406.2

# Install pip dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

RUN MAX_JOBS=4 pip install flash-attn==2.5.9.post1 --no-build-isolation

# Inference requirements
COPY --from=mcr.microsoft.com/azureml/o16n-base/python-assets:20230419.v1 /artifacts /var/
RUN /var/requirements/install_system_requirements.sh && \\
    cp /var/configuration/rsyslog.conf /etc/rsyslog.conf && \\
    cp /var/configuration/nginx.conf /etc/nginx/sites-available/app && \\
    ln -sf /etc/nginx/sites-available/app /etc/nginx/sites-enabled/app && \\
    rm -f /etc/nginx/sites-enabled/default
ENV SVDIR=/var/runit
ENV WORKER_TIMEOUT=400
EXPOSE 5001 8883 8888

# support Deepspeed launcher requirement of passwordless ssh login
RUN apt-get update
RUN apt-get install -y openssh-server openssh-client
```

### 3.4. Create an Endpoint <a href="#community-4181123-toc-hid-773209900" id="community-4181123-toc-hid-773209900"></a>

An endpoint refers to an HTTP(S) URL that makes the model accessible from the outside. Endpoint can have multiple deployments, which allows traffic to be distributed across multiple deployments. Endpoint does the following:

1. **API interface provided**: Endpoint provides a URL to receive model prediction requests through a RESTful API.
2. **Traffic routing**: Endpoint distributes traffic across multiple deployments. This allows you to implement A/B testing or canary deployment strategies.
3. **Scalability**: Endpoint supports scaling across multiple deployments and can be load balanced across additional deployments as traffic increases.
4. **Security Management**: Endpoints secure models through authentication and authorization. You can control access using API keys or Microsoft Entra ID.

The code snippet is below. Note that this process does not provision a compute cluster yet.

```applescript
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    IdentityConfiguration,
    ManagedIdentityConfiguration,
)

# Check if the endpoint already exists in the workspace
try:
    endpoint = ml_client.online_endpoints.get(azure_endpoint_name)
    print("---Endpoint already exists---")
except:
    # Create an online endpoint if it doesn't exist
    endpoint = ManagedOnlineEndpoint(
        name=azure_endpoint_name,
        description=f"Test endpoint for {model.name}",
    )

# Trigger the endpoint creation
try:
    ml_client.begin_create_or_update(endpoint).wait()
    print("\\n---Endpoint created successfully---\\n")
except Exception as err:
    raise RuntimeError(
        f"Endpoint creation failed. Detailed Response:\\n{err}"
    ) from err
```

### 3.5. Create a Deployment <a href="#community-4181123-toc-hid-1714302933" id="community-4181123-toc-hid-1714302933"></a>

Deployment is the instance that actually run the model. Multiple deployments can be connected to an endpoint, and each deployment contains a model, environment, compute resources, infrastructure settings, and more. Deployment does the following:

1. **Manage resources**: The deployment manages the computing resources needed to run the model. You can set up resources like CPU, GPU, and memory.
2. **Versioning**: Deployments can manage different versions of a model. This makes it easy to roll back to a previous version or deploy a new version.
3. **Monitoring and logging**: We can monitor the logs and performance of running models. This helps you detect and resolve issues.

The code snippet is below. Note that this takes a lot of time as a GPU cluster must be provisioned and the serving environment must be built.

```applescript
from azure.ai.ml.entities import (    
    OnlineRequestSettings,
    CodeConfiguration,
    ManagedOnlineDeployment,
    ProbeSettings,
    Environment
)

deployment = ManagedOnlineDeployment(
    name=azure_deployment_name,
    endpoint_name=azure_endpoint_name,
    model=model,
    instance_type=azure_serving_cluster_size,
    instance_count=1,
    #code_configuration=code_configuration,
    environment = env,
    scoring_script="score.py",
    code_path="./src_serve",
    #environment_variables=deployment_env_vars,
    request_settings=OnlineRequestSettings(max_concurrent_requests_per_instance=3,
                                           request_timeout_ms=90000, max_queue_wait_ms=60000),
    liveness_probe=ProbeSettings(
        failure_threshold=30,
        success_threshold=1,
        period=100,
        initial_delay=500,
    ),
    readiness_probe=ProbeSettings(
        failure_threshold=30,
        success_threshold=1,
        period=100,
        initial_delay=500,
    ),
)

# Trigger the deployment creation
try:
    ml_client.begin_create_or_update(deployment).wait()
    print("\\n---Deployment created successfully---\\n")
except Exception as err:
    raise RuntimeError(
        f"Deployment creation failed. Detailed Response:\\n{err}"
    ) from err
    
endpoint.traffic = {azure_deployment_name: 100}
endpoint_poller = ml_client.online_endpoints.begin_create_or_update(endpoint)   
```

> **\[Tip]** Please specify and deploy the liveness probe settings directly to check if the model deployment container is running normally. When debugging, it is recommended to set a high initial\_delay and a high failure\_threshold and high period for error log analysis. Please check `ProbeSettings()` in the code above.

\
4\. Invocation <a href="#community-4181123-toc-hid-351316968" id="community-4181123-toc-hid-351316968"></a>
-----------------------------------------------------------------------------------------------------------

***

We finally succeeded in serving the Florence-2 model. Try using the code below to perform model inference.

```applescript
import os
import json
import base64

with open('./DocumentVQA_Test_01.jpg', 'rb') as img:
    base64_img = base64.b64encode(img.read()).decode('utf-8')
    
sample = {
    "task_prompt": "DocVQA",
    "image_input": base64_img,
    "text_input": "What do you see in this image", 
    "params": {
        "max_new_tokens": 512,
        "num_beams": 3
    }
}

test_src_dir = "./inference-test"
os.makedirs(test_src_dir, exist_ok=True)
print(f"test script directory: {test_src_dir}")
sample_data_path = os.path.join(test_src_dir, "sample-request.json")

with open(sample_data_path, "w") as f:
    json.dump(sample, f)
result = ml_client.online_endpoints.invoke(
    endpoint_name=azure_endpoint_name,
    deployment_name=azure_deployment_name,
    request_file=sample_data_path,
)

result_json = json.loads(result)
print(result_json['result'])
```

It is a good strategy to perform LLM latency/throughput benchmarking before deploying the model in earnest. Benchmark the following metrics as a baseline.

```applescript
metrics = {
    'threads': num_threads,
    'duration': duration,
    'throughput': throughput,
    'avg_sec': avg_latency,
    'std_sec': time_std_sec,        
    'p95_sec': time_p95_sec,
    'p99_sec': time_p99_sec    
}
```

\
We have published the code to do this post end-to-end at [https://github.com/Azure/azure-llm-fine-tuning/tree/main/florence2-VQA](https://github.com/Azure/azure-llm-fine-tuning/tree/main/florence2-VQA).

We hope this tutorial will help you fine-tune and deploy modern models, including the Florence-2 model, in Azure ML Studio.

## References <a href="#community-4181123-toc-hid-2136195865" id="community-4181123-toc-hid-2136195865"></a>

* [Hugging Face blog - Fine-tuning Florence-2](https://huggingface.co/blog/finetune-florence2)
* [Fine-tune SLM Phi-3 using Azure ML](https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/finetune-small-language-model-slm-phi-3-using-azure-machine/ba-p/4130399)
* [Hands-on labs - LLM Fine-tuning/serving with Azure ML](https://github.com/Azure/azure-llm-fine-tuning) &#x20;
