#!/bin/bash

set -euo pipefail

# Color codes for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Get script directory
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ENV_FILE="${SCRIPT_DIR}/.env"

# Show usage information
show_usage() {
    echo "Usage: $0 [--up|--down|--help]"
    echo ""
    echo "Options:"
    echo "  --up      Deploy the model endpoint and deployment"
    echo "  --down    Clean up and delete the endpoint and deployment"
    echo "  --help    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --up      # Deploy the model"
    echo "  $0 --down    # Clean up resources"
}

# Load environment variables
load_env_vars() {
    if [[ ! -f "$ENV_FILE" ]]; then
        log_error ".env file not found at $ENV_FILE"
        if [[ "$1" == "deploy" ]]; then
            log_info "Please copy .env.example to .env and fill in your values"
        fi
        exit 1
    fi

    log_info "Loading environment variables from .env file..."
    
    while IFS= read -r line || [[ -n "$line" ]]; do
        if [[ "$line" =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
            continue
        fi
        
        if [[ "$line" =~ ^[[:alpha:]_][[:alnum:]_]*= ]]; then
            export "$line"
            if [[ "$1" == "deploy" ]]; then
                log_info "Loaded: $(echo "$line" | cut -d'=' -f1)"
            fi
        fi
    done < "$ENV_FILE"

    if [[ -z "${WORKSPACE:-}" ]] || [[ -z "${RESOURCE_GROUP:-}" ]]; then
        log_error "Required environment variables WORKSPACE and RESOURCE_GROUP must be set in .env file"
        exit 1
    fi
}

# Check if Azure CLI is installed and logged in
check_azure_cli() {
    log_info "Checking Azure CLI..."
    
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi

    if ! az account show &> /dev/null; then
        log_info "Not logged in to Azure. Initiating login..."
        az login
    else
        log_success "Already logged in to Azure"
        local account_name
        account_name=$(az account show --query "name" -o tsv)
        log_info "Current account: $account_name"
    fi
}

# Verify Azure ML workspace exists
verify_workspace() {
    log_info "Verifying Azure ML workspace access..."
    
    if ! az ml workspace show -n "$WORKSPACE" -g "$RESOURCE_GROUP" &> /dev/null; then
        log_error "Workspace '$WORKSPACE' not found in resource group '$RESOURCE_GROUP'"
        log_info "Available workspaces:"
        az ml workspace list -g "$RESOURCE_GROUP" --query "[].name" -o table || true
        exit 1
    fi

    log_success "Workspace '$WORKSPACE' verified in resource group '$RESOURCE_GROUP'"
}

# Deploy endpoint
deploy_endpoint() {
    local endpoint_name
    endpoint_name=$(grep "^name:" "${SCRIPT_DIR}/endpoint.yml" | cut -d' ' -f2)
    
    log_info "Deploying endpoint '$endpoint_name'..."
    
    if az ml online-endpoint show -n "$endpoint_name" -g "$RESOURCE_GROUP" -w "$WORKSPACE" &> /dev/null; then
        log_warning "Endpoint '$endpoint_name' already exists, skipping creation"
    else
        log_info "Creating new endpoint '$endpoint_name'..."
        az ml online-endpoint create -f "${SCRIPT_DIR}/endpoint.yml" -g "$RESOURCE_GROUP" -w "$WORKSPACE"
        log_success "Endpoint '$endpoint_name' created successfully"
    fi
}

# Deploy deployment
deploy_deployment() {
    local endpoint_name deployment_name
    endpoint_name=$(grep "^endpoint_name:" "${SCRIPT_DIR}/deployment.yml" | cut -d' ' -f2)
    deployment_name=$(grep "^name:" "${SCRIPT_DIR}/deployment.yml" | cut -d' ' -f2)
    
    log_info "Deploying '$deployment_name' deployment to endpoint '$endpoint_name'..."
    
    if az ml online-deployment show -n "$deployment_name" -e "$endpoint_name" -g "$RESOURCE_GROUP" -w "$WORKSPACE" &> /dev/null; then
        log_warning "Deployment '$deployment_name' already exists, updating..."
        az ml online-deployment update -f "${SCRIPT_DIR}/deployment.yml" -g "$RESOURCE_GROUP" -w "$WORKSPACE"
    else
        log_info "Creating new deployment '$deployment_name'..."
        az ml online-deployment create -f "${SCRIPT_DIR}/deployment.yml" -g "$RESOURCE_GROUP" -w "$WORKSPACE"
    fi
    
    log_success "Deployment '$deployment_name' completed"
    
    log_info "Setting traffic to 100% for deployment '$deployment_name'..."
    az ml online-endpoint update -n "$endpoint_name" --traffic "$deployment_name=100" -g "$RESOURCE_GROUP" -w "$WORKSPACE"
    log_success "Traffic updated successfully"
}

# Monitor deployment logs
monitor_deployment() {
    local endpoint_name deployment_name
    endpoint_name=$(grep "^endpoint_name:" "${SCRIPT_DIR}/deployment.yml" | cut -d' ' -f2)
    deployment_name=$(grep "^name:" "${SCRIPT_DIR}/deployment.yml" | cut -d' ' -f2)
    
    log_info "Waiting for deployment to be ready..."
    
    local max_attempts=60
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        local state
        state=$(az ml online-deployment show -n "$deployment_name" -e "$endpoint_name" -g "$RESOURCE_GROUP" -w "$WORKSPACE" --query "provisioning_state" -o tsv)
        
        case "$state" in
            "Succeeded")
                log_success "Deployment is ready!"
                break
                ;;
            "Failed")
                log_error "Deployment failed!"
                az ml online-deployment get-logs -n "$deployment_name" -e "$endpoint_name" -g "$RESOURCE_GROUP" -w "$WORKSPACE" --lines 50
                exit 1
                ;;
            "Canceled")
                log_error "Deployment was canceled!"
                exit 1
                ;;
        esac
        
        log_info "Deployment state: $state (attempt $((attempt + 1))/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        log_error "Timeout waiting for deployment to be ready"
        exit 1
    fi
    
    log_info "Showing recent deployment logs..."
    az ml online-deployment get-logs -n "$deployment_name" -e "$endpoint_name" -g "$RESOURCE_GROUP" -w "$WORKSPACE" --lines 100
    
    log_info "Endpoint details:"
    az ml online-endpoint show -n "$endpoint_name" -g "$RESOURCE_GROUP" -w "$WORKSPACE" --query "{name:name,scoring_uri:scoring_uri,state:provisioning_state}" -o table
}

# Get and display API key
get_api_key() {
    local endpoint_name
    endpoint_name=$(grep "^name:" "${SCRIPT_DIR}/endpoint.yml" | cut -d' ' -f2)
    
    log_info "Retrieving API key for endpoint '$endpoint_name'..."
    
    local api_key
    if api_key=$(az ml online-endpoint get-credentials -n "$endpoint_name" -g "$RESOURCE_GROUP" -w "$WORKSPACE" --query "primaryKey" -o tsv 2>/dev/null); then
        log_success "API key retrieved successfully!"
        echo ""
        echo "=== ENDPOINT CREDENTIALS ==="
        echo "Endpoint Name: $endpoint_name"
        echo "API Key: $api_key"
        echo "============================"
        echo ""
        log_info "Save this API key securely - you'll need it to make requests to your endpoint"
    else
        log_warning "Could not retrieve API key. You can get it later with:"
        log_info "az ml online-endpoint get-credentials -n $endpoint_name -g $RESOURCE_GROUP -w $WORKSPACE"
    fi
}

# Delete endpoint (which implicitly deletes deployments)
cleanup_resources() {
    local endpoint_name
    endpoint_name=$(grep "^name:" "${SCRIPT_DIR}/endpoint.yml" | cut -d' ' -f2)
    
    log_info "Cleaning up Azure ML resources..."
    
    if az ml online-endpoint show -n "$endpoint_name" -g "$RESOURCE_GROUP" -w "$WORKSPACE" &> /dev/null; then
        log_info "Deleting endpoint '$endpoint_name' (this will also delete associated deployments)..."
        az ml online-endpoint delete -n "$endpoint_name" -g "$RESOURCE_GROUP" -w "$WORKSPACE" --yes
        log_success "Endpoint '$endpoint_name' and associated deployments deleted"
    else
        log_warning "Endpoint '$endpoint_name' not found"
    fi
}

# Deploy function
deploy() {
    log_info "Starting Azure ML endpoint deployment..."
    log_info "Script directory: $SCRIPT_DIR"
    
    load_env_vars "deploy"
    check_azure_cli
    verify_workspace
    deploy_endpoint
    deploy_deployment
    monitor_deployment
    get_api_key
    
    log_success "Deployment completed successfully!"
    log_info "Your endpoint is now ready to serve requests."
}

# Cleanup function
cleanup() {
    log_info "Starting cleanup of Azure ML resources..."
    
    load_env_vars "cleanup"
    cleanup_resources
    
    log_success "Cleanup completed successfully!"
}

# Main execution
main() {
    if [[ $# -eq 0 ]]; then
        log_error "No arguments provided"
        show_usage
        exit 1
    fi

    case "$1" in
        --up)
            deploy
            ;;
        --down)
            cleanup
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
