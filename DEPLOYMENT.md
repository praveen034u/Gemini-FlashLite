# Cloud Run Deployment Guide

## Prerequisites

### 1. Google Cloud Project Setup
- Create a GCP project or use an existing one
- Enable required APIs:
  ```bash
  gcloud services enable run.googleapis.com
  gcloud services enable artifactregistry.googleapis.com
  gcloud services enable cloudbuild.googleapis.com
  ```

### 2. Create Artifact Registry Repository
```bash
gcloud artifacts repositories create ai-chat-api-repo \
  --repository-format=docker \
  --location=us-central1 \
  --description="AI Chat API Docker repository"
```

### 3. Create Service Account
```bash
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:github-actions-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:github-actions-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:github-actions-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Create and download key
gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=github-actions-sa@PROJECT_ID.iam.gserviceaccount.com
```

## GitHub Secrets Configuration

Add the following secrets in your GitHub repository (Settings > Secrets and variables > Actions):

### Required Secrets:
1. **GCP_SA_KEY**: Content of `github-actions-key.json`
2. **GCP_PROJECT_ID**: Your GCP project ID (e.g., `my-project-123`)
3. **REGION**: Deployment region (e.g., `us-central1`)
4. **SERVICE_NAME**: Cloud Run service name (e.g., `ai-chat-api`)
5. **POSTGRES_URL**: PostgreSQL connection string from Supabase
   ```
   postgresql://USER:PASSWORD@HOST:6543/postgres?pgbouncer=true
   ```
6. **GOOGLE_API_KEY**: Google Gemini API key
7. **OPENAI_API_KEY**: OpenAI API key

### Optional Secrets (for Secret Manager integration):
If using Google Secret Manager instead of environment variables:
```bash
# Create secrets in Secret Manager
echo -n "your-postgres-url" | gcloud secrets create POSTGRES_URL --data-file=-
echo -n "your-google-api-key" | gcloud secrets create GOOGLE_API_KEY --data-file=-
echo -n "your-openai-api-key" | gcloud secrets create OPENAI_API_KEY --data-file=-

# Grant service account access to secrets
gcloud secrets add-iam-policy-binding POSTGRES_URL \
  --member="serviceAccount:github-actions-sa@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Local Testing with Docker

### Build the image locally:
```bash
docker build -t ai-chat-api:local .
```

### Run locally with environment variables:
```bash
docker run -p 8080:8080 \
  -e POSTGRES_URL="postgresql://..." \
  -e GOOGLE_API_KEY="your-key" \
  -e OPENAI_API_KEY="your-key" \
  ai-chat-api:local
```

### Test the API:
```bash
curl http://localhost:8080/docs
```

## Manual Deployment to Cloud Run

If you prefer manual deployment instead of CI/CD:

```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project PROJECT_ID

# Build and push image
docker build -t us-central1-docker.pkg.dev/PROJECT_ID/ai-chat-api-repo/ai-chat-api:latest .
docker push us-central1-docker.pkg.dev/PROJECT_ID/ai-chat-api-repo/ai-chat-api:latest

# Deploy to Cloud Run
gcloud run deploy ai-chat-api \
  --image us-central1-docker.pkg.dev/PROJECT_ID/ai-chat-api-repo/ai-chat-api:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --cpu 1 \
  --memory 512Mi \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 900s \
  --set-env-vars "POSTGRES_URL=...,GOOGLE_API_KEY=...,OPENAI_API_KEY=..."
```

## CI/CD Workflow

The GitHub Actions workflow (`.github/workflows/docker-image.yml`) automatically:

1. ✅ Builds Docker image on push to `main` branch
2. ✅ Pushes image to Artifact Registry
3. ✅ Deploys to Cloud Run with all environment variables
4. ✅ Tags images with commit SHA and `latest`
5. ✅ Outputs the deployed service URL

### Trigger Deployment:
- **Automatic**: Push to `main` branch
- **Manual**: Go to Actions tab → Select workflow → Run workflow

## Configuration Files

### Dockerfile
- Uses Python 3.11 slim base image
- Installs system dependencies (gcc, libpq-dev)
- Installs Python packages from requirements.txt
- Runs as non-root user for security
- Exposes port 8080
- Includes health check

### requirements.txt
All dependencies with pinned versions for reproducible builds:
- FastAPI & Uvicorn
- Database (psycopg2-binary, SQLAlchemy)
- LLM providers (openai, google-genai)
- LangChain
- Utilities

### llm_config.json
Configuration for:
- Default LLM provider
- History limits
- LLM settings per provider
- Role-based system prompts and rules

## Environment Variables

### Production Environment Variables:
```bash
# Database (Required)
POSTGRES_URL=postgresql://...

# LLM API Keys (At least one required)
GOOGLE_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-...

# Optional
PORT=8080  # Cloud Run sets this automatically
```

## Monitoring and Logs

### View logs:
```bash
gcloud run services logs read ai-chat-api \
  --region us-central1 \
  --limit 50
```

### Monitor metrics:
```bash
gcloud run services describe ai-chat-api \
  --region us-central1 \
  --format="value(status.url)"
```

### Cloud Console:
- Visit: https://console.cloud.google.com/run
- Select your service to view metrics, logs, and configuration

## Scaling Configuration

Current settings in workflow:
- **CPU**: 1 vCPU
- **Memory**: 512 Mi
- **Min Instances**: 0 (scales to zero)
- **Max Instances**: 10
- **Concurrency**: 80 requests per instance
- **Timeout**: 900 seconds (15 minutes)

To adjust:
```bash
gcloud run services update ai-chat-api \
  --region us-central1 \
  --memory 1Gi \
  --cpu 2 \
  --max-instances 20
```

## Cost Optimization

- ✅ Scales to zero when not in use
- ✅ Minimal docker image size
- ✅ Efficient caching with `.dockerignore`
- ✅ Container security with non-root user
- ✅ Background tasks for non-blocking operations

## Security Best Practices

1. ✅ Service account with minimal permissions
2. ✅ Secrets stored in GitHub Secrets (or Secret Manager)
3. ✅ Non-root container user
4. ✅ Pinned dependency versions
5. ✅ HTTPS enabled by default on Cloud Run
6. ⚠️ Currently allows unauthenticated access (adjust if needed)

## Troubleshooting

### Build fails in GitHub Actions:
- Check GitHub Secrets are properly set
- Verify service account has required permissions
- Check Artifact Registry repository exists

### Deployment fails:
- Verify environment variables are correctly set
- Check Cloud Run service account permissions
- Review deployment logs in GitHub Actions

### API not responding:
- Check logs: `gcloud run services logs read SERVICE_NAME`
- Verify environment variables are set in Cloud Run
- Test database connectivity
- Verify API keys are valid

### Cold start issues:
- Consider setting `--min-instances 1` for production
- Optimize application startup time
- Use Cloud Run warmup requests

## Support

For issues or questions:
1. Check Cloud Run logs
2. Review GitHub Actions workflow logs
3. Verify all secrets and configurations
4. Test locally with Docker first
