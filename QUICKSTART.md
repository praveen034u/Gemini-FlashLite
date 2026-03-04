# Quick Start: Cloud Run Deployment

## 🚀 Fast Track Deployment (5 minutes)

### 1️⃣ GCP Setup (One-time)
```bash
# Set your project ID
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export SERVICE_NAME="ai-chat-api"

# Enable APIs
gcloud services enable run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

# Create Artifact Registry
gcloud artifacts repositories create ai-chat-api-repo \
  --repository-format=docker \
  --location=$REGION

# Create service account
gcloud iam service-accounts create github-actions-sa
```

### 2️⃣ Grant Permissions
```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

### 3️⃣ Create Service Account Key
```bash
gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=github-actions-sa@$PROJECT_ID.iam.gserviceaccount.com

# Display key (copy this for GitHub)
cat github-actions-key.json
```

### 4️⃣ Configure GitHub Secrets

Go to: Repository → Settings → Secrets and variables → Actions → New repository secret

Add these 7 secrets:

| Secret Name | Example Value | Description |
|------------|---------------|-------------|
| `GCP_SA_KEY` | `{...json content...}` | Content from github-actions-key.json |
| `GCP_PROJECT_ID` | `my-project-123` | Your GCP project ID |
| `REGION` | `us-central1` | GCP region |
| `SERVICE_NAME` | `ai-chat-api` | Your service name |
| `POSTGRES_URL` | `postgresql://user:pass@host:6543/db` | Supabase connection string |
| `GOOGLE_API_KEY` | `AIzaSy...` | Google Gemini API key |
| `OPENAI_API_KEY` | `sk-...` | OpenAI API key |

### 5️⃣ Deploy
```bash
# Push to main branch
git add .
git commit -m "Configure Cloud Run deployment"
git push origin main

# Or manually trigger in GitHub Actions
# Go to: Actions → CI/CD to Cloud Run → Run workflow
```

### 6️⃣ Verify
```bash
# Get service URL
gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --format='value(status.url)'

# Test API (replace URL with yours)
curl https://ai-chat-api-xxxxx-uc.a.run.app/docs
```

## 📦 Manual Local Build & Deploy

### Build Docker Image
```bash
docker build -t ai-chat-api:latest .
```

### Run Locally
```bash
docker run -p 8080:8080 \
  -e POSTGRES_URL="postgresql://..." \
  -e GOOGLE_API_KEY="AIzaSy..." \
  -e OPENAI_API_KEY="sk-..." \
  ai-chat-api:latest
```

### Test Locally
```bash
curl http://localhost:8080/docs
```

### Push and Deploy
```bash
# Tag image
docker tag ai-chat-api:latest \
  $REGION-docker.pkg.dev/$PROJECT_ID/ai-chat-api-repo/ai-chat-api:latest

# Authenticate Docker
gcloud auth configure-docker $REGION-docker.pkg.dev

# Push image
docker push $REGION-docker.pkg.dev/$PROJECT_ID/ai-chat-api-repo/ai-chat-api:latest

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/ai-chat-api-repo/ai-chat-api:latest \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 512Mi \
  --set-env-vars "POSTGRES_URL=$POSTGRES_URL,GOOGLE_API_KEY=$GOOGLE_API_KEY,OPENAI_API_KEY=$OPENAI_API_KEY"
```

## 🧪 Test Deployed API

### Using curl:
```bash
# Get your service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --format='value(status.url)')

# Test generate endpoint
curl -X POST "$SERVICE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "prompt": "What is 2+2?",
    "role": "student"
  }'

# Test history endpoint
curl "$SERVICE_URL/history/test_user"
```

### Using Python:
```python
import requests

SERVICE_URL = "https://your-service-url.a.run.app"

# Test generate
response = requests.post(f"{SERVICE_URL}/generate", json={
    "user_id": "test_user",
    "prompt": "Explain photosynthesis",
    "role": "student"
})
print(response.json())

# Test history
history = requests.get(f"{SERVICE_URL}/history/test_user")
print(history.json())
```

## 📊 Monitor Deployment

### View Logs:
```bash
gcloud run services logs read $SERVICE_NAME \
  --region=$REGION \
  --limit=50
```

### Check Status:
```bash
gcloud run services describe $SERVICE_NAME --region=$REGION
```

### View in Console:
https://console.cloud.google.com/run

## 🔧 Update Configuration

### Update Environment Variables:
```bash
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --set-env-vars "NEW_VAR=value"
```

### Update Scaling:
```bash
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --memory=1Gi \
  --cpu=2 \
  --max-instances=20 \
  --min-instances=1
```

## 🆘 Troubleshooting

### Check Build Logs:
- Go to GitHub → Actions → Select workflow run
- Review each step for errors

### Check Deployment Logs:
```bash
gcloud run services logs read $SERVICE_NAME --region=$REGION
```

### Common Issues:

**"Permission denied"**
- Verify service account has required roles
- Check IAM permissions

**"Image not found"**
- Verify Artifact Registry name matches workflow
- Check image was pushed successfully

**"Service unavailable"**
- Check environment variables are set
- Verify database connection
- Review application logs

**"Out of memory"**
- Increase memory: `--memory=1Gi`
- Optimize application code

## 📚 Next Steps

- ✅ Review [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guide
- ✅ Check [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) for complete checklist
- ✅ Set up monitoring and alerts
- ✅ Configure custom domain (optional)
- ✅ Enable Cloud CDN (optional)
- ✅ Set up staging environment

## 💡 Pro Tips

1. **Test locally first**: Always build and test Docker image locally before pushing
2. **Use Secret Manager**: For production, use Google Secret Manager instead of env vars
3. **Enable Cloud Armor**: Add DDoS protection for production
4. **Set up Budget Alerts**: Monitor Cloud Run costs
5. **Use Cloud Trace**: Track request latency and performance
6. **Implement Health Checks**: Already included in Dockerfile
7. **Version your images**: Use git SHA tags (automatically done in workflow)

## 🎉 That's it!

Your AI Chat API should now be deployed and running on Cloud Run!

Service URL format: `https://[SERVICE_NAME]-[HASH]-[REGION].a.run.app`

---

Need help? Check:
- [Full Deployment Guide](DEPLOYMENT.md)
- [Deployment Checklist](DEPLOYMENT_CHECKLIST.md)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
