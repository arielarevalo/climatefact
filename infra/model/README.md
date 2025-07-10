# Azure ML Model Deployment

This directory contains the configuration and scripts needed to deploy the Microsoft DeBERTa Large MNLI model to Azure Machine Learning as a managed online endpoint.

## Overview

The deployment uses the HuggingFace model registry to deploy `microsoft/deberta-large-mnli`, a natural language inference model that can classify the relationship between pairs of sentences.

## Files Structure

```
model/
├── README.md                 # This file
├── .env.example             # Environment variables template
├── .env                     # Your environment variables (not in git)
├── conda.yml                # Conda environment specification
├── endpoint.yml             # Azure ML endpoint configuration
├── deployment.yml           # Azure ML deployment configuration
├── up.sh                    # Deployment script
└── down.sh                  # Cleanup script
```

## Prerequisites

1. **Azure CLI**: Install and configure the Azure CLI
   ```bash
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
   ```

2. **Azure ML CLI Extension**: Install the Azure ML extension
   ```bash
   az extension add -n ml
   ```

3. **Azure Subscription**: You need an active Azure subscription with:
   - An Azure ML workspace
   - Appropriate permissions to create endpoints and deployments

## Setup

1. **Configure Environment Variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your Azure details:
   ```bash
   WORKSPACE=your-workspace-name
   RESOURCE_GROUP=your-resource-group
   ```

2. **Make Script Executable**:
   ```bash
   chmod +x manage_deployment.sh
   ```

## Usage

### Deploy the Model

Run the deployment script to create the endpoint and deployment:

```bash
./manage_deployment.sh --up
```

This script will:
- Load environment variables from `.env`
- Check Azure CLI authentication
- Verify workspace access
- Create the endpoint (if it doesn't exist)
- Deploy the model
- Set traffic routing to 100%
- Monitor deployment status
- Show deployment logs
- Retrieve and display the API key

### Test the Deployment

Once deployed, you can test the endpoint using the Azure CLI:

```bash
az ml online-endpoint invoke \
  -n deberta-large-mnli \
  -g your-resource-group \
  -w your-workspace \
  --request-file test_data.json
```

Or using curl with the API key (displayed after deployment):

```bash
curl -X POST "https://your-endpoint-uri.region.inference.ml.azure.com/score" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

Example test data (`test_data.json`):
```json
{
  "inputs": {
    "premise": "A person on a horse jumps over a broken down airplane.",
    "hypothesis": "A person is outdoors, on a horse."
  }
}
```

### Clean Up Resources

To delete the deployment and endpoint:

```bash
./manage_deployment.sh --down
```

This will remove both the deployment and endpoint to avoid ongoing costs.

### Get Help

To see available options:

```bash
./manage_deployment.sh --help
```

## Configuration Details

### Model Information
- **Model**: `microsoft/deberta-large-mnli`
- **Task**: Natural Language Inference (NLI)
- **Framework**: Transformers
- **Instance Type**: Standard_E4s_v3 (4 vCPUs, 32 GB RAM)

### Environment
- **Base Image**: `mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest`
- **Python**: 3.10
- **Key Dependencies**: PyTorch, Transformers, Accelerate

### Environment Variables
- `HF_HUB_ENABLE_HF_TRANSFER="1"`: Enables faster model downloads
- `TRANSFORMERS_NO_ADVISORY_WARNINGS="1"`: Reduces warning noise

## Troubleshooting

### Common Issues

1. **Authentication Errors**:
   ```bash
   az login
   az account set --subscription "your-subscription-id"
   ```

2. **Workspace Not Found**:
   - Verify your workspace name and resource group in `.env`
   - Check you have access to the workspace

3. **Deployment Failures**:
   - Check logs: `az ml online-deployment get-logs -n main -e deberta-large-mnli -g your-rg -w your-workspace`
   - Verify compute quotas in your subscription

4. **Script Permission Denied**:
   ```bash
   chmod +x manage_deployment.sh
   ```

### Monitoring

View real-time logs during deployment:
```bash
az ml online-deployment get-logs -n main -e deberta-large-mnli -g your-rg -w your-workspace --lines 100 -f
```

Check endpoint status:
```bash
az ml online-endpoint show -n deberta-large-mnli -g your-rg -w your-workspace
```

## Cost Considerations

- The Standard_E4s_v3 instance costs approximately $0.31/hour
- Remember to run `./down.sh` when not using the endpoint
- Consider using smaller instance types for development/testing

## Security

- The endpoint uses key-based authentication by default
- The API key is displayed after successful deployment
- Retrieve the key later: `az ml online-endpoint get-credentials -n deberta-large-mnli -g your-rg -w your-workspace`
- Consider using Azure AD authentication for production

## Support

For issues related to:
- Azure ML: Check [Azure ML documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- Model performance: See [HuggingFace model card](https://huggingface.co/microsoft/deberta-large-mnli)
- Scripts: Review logs and error messages in the terminal output
