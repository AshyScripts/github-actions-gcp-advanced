# CI/CD Pipeline with GitHub Actions and GCP
## Introduction
This lab guides you through building a continuous integration and continuous deployment (CI/CD) pipeline using GitHub Actions and Google Cloud Platform (GCP) services. You will learn how to automate data processing, model evaluation, retraining, and deployment using services like Google Cloud Storage (GCS), Cloud Functions, and Artifact Registry.

## Learning objectives
By completing this lab, you will:

- Set Up a CI/CD Pipeline with GitHub Actions: Learn how to automate workflows using GitHub Actions to streamline the development process.
- Use Google Cloud Storage for Data Handling: Understand how to store, retrieve, and manage data using GCS buckets.
- Automate Model Retraining with Cloud Functions: Discover how to trigger serverless functions to automate the retraining of machine learning models.
- Build and Push Docker Images to Artifact Registry: Gain experience in containerizing applications and managing them with GCP's Artifact Registry.

## Structure of pipeline
The pipeline consists of several components that work together to automate the machine learning workflow:

- Data Processing (`data_processing.py`): Downloads raw data from GCS, introduces data drift to simulate real-world changes, preprocesses the data, and uploads the processed data back to GCS.
- Model Evaluation (`evaluate_model.py`): Downloads the latest model and processed data from GCS, evaluates the model's performance, and determines if retraining is necessary based on a predefined accuracy threshold.
- Model Training (`train_model.py`): Retrains the model using the processed data and uploads the new model to GCS.
- CI/CD Pipeline (`ci_cd.yaml`): A GitHub Actions workflow that orchestrates the entire process, including data processing, model evaluation, conditional retraining, and building and pushing a Docker image to Artifact Registry.
- Cloud Function (`retrain_model`): A serverless function on GCP that automates the retraining process when triggered.

## Evaluating the model and deciding on retraining
fter the data processing step, the pipeline evaluates the current machine learning model using the `evaluate_model.py` script. This evaluation determines whether the model meets a predefined accuracy threshold and decides if retraining is necessary.
In the GitHub Actions workflow (`ci_cd.yaml`), the evaluation step is defined as:
```yaml
- name: Evaluate Model
  id: evaluate
  run: |
    echo "Starting model evaluation..."
    # Remove any existing model file
    rm -f /tmp/model.pkl

    set +e  # Disable immediate exit on error
    python evaluate_model.py
    EXIT_CODE=$?
    set -e  # Re-enable immediate exit on error

    if [ $EXIT_CODE -ne 0 ]; then
      echo "Evaluation failed with exit code $EXIT_CODE"
      echo "retrain=true" >> $GITHUB_OUTPUT
    else
      echo "Model accuracy meets threshold - no retraining needed"
      echo "retrain=false" >> $GITHUB_OUTPUT
    fi

    # Exit with zero to prevent the step from failing
    exit 0
```

How It Works:

The pipeline executes python `evaluate_model.py`, which assesses the model's performance on the processed data. The script exits with a status code:

- 0 if the model meets or exceeds the accuracy threshold.
- Non-zero if the model fails to meet the threshold or encounters an error.
- Capturing the Exit Code: The set +e command temporarily disables the shell's default behavior of exiting immediately on error (set -e is enabled by default in GitHub Actions). This allows the script to capture the exit code of evaluate_model.py without stopping the entire workflow.

- Determining Retraining Need: The script checks the exit code stored in the `EXIT_CODE` variable. If the exit code is not zero (`EXIT_CODE != 0`), it indicates that the model's performance is insufficient, and retraining is required.

Setting an Output Variable: Based on the exit code, the script sets an output variable retrain using:
```bash
echo "retrain=true" >> $GITHUB_OUTPUT
```
This output variable is then accessible in subsequent steps to conditionally trigger actions based on whether retraining is needed.

## Triggering Retraining via Cloud Function
If the evaluation step determines that retraining is necessary, the pipeline triggers a Google Cloud Function that handles the retraining process. This is done using the following step:

```yaml
- name: Trigger Retraining
  if: steps.evaluate.outputs.retrain == 'true'
  run: |
    echo "Starting retraining process..."
    ACCESS_TOKEN=$(gcloud auth print-identity-token)
    curl -X POST \
      -H "Authorization: Bearer $ACCESS_TOKEN" \
      -H "Content-Type: application/json" \
      --max-time 30 \
      --verbose \
      https://us-east4-project-demo-1-439713.cloudfunctions.net/retrain_model
```
Here's how this part is working:
- Obtaining an Identity Token: 
```bash
ACCESS_TOKEN=$(gcloud auth print-identity-token)
```
retrieves an identity token using the gcloud CLI. This token is used to authenticate the request to the Cloud Function, ensuring that only authorized clients can trigger it.
- Triggering the Cloud Function: The `curl` command sends an HTTP POST request to the Cloud Function's URL to initiate the retraining process.

Understanding curl Options:

- `-X POST`: Specifies the HTTP method as POST.
- `-H "Authorization: Bearer $ACCESS_TOKEN"`: Adds an Authorization header with the bearer token for authentication.
- `-H "Content-Type: application/json"`: Sets the Content-Type header to indicate the request body format (even if the body is empty).
- `--max-time 30`: Limits the maximum time for the request to 30 seconds to prevent hanging.

## Cloud Function for Automated Model Retraining
The Cloud Function `retrain_model` serves as the orchestrator for automating the model retraining process. When triggered—typically via an HTTP POST request from the CI/CD pipeline—it initiates the retraining workflow by executing two key scripts: `data_processing.py` and `train_model.py`. This is achieved using Python's `subprocess.run` method within main.py, allowing the Cloud Function to run these scripts as separate subprocesses while capturing their outputs and handling any errors.


