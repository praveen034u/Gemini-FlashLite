# Cloud Run Deployment Checklist

## 📋 Pre-Deployment Setup

### GCP Configuration
- [ ] GCP project created/selected
- [ ] Billing enabled on project
- [ ] Required APIs enabled:
  - [ ] Cloud Run API
  - [ ] Artifact Registry API
  - [ ] Cloud Build API
- [ ] Artifact Registry repository created: `ai-chat-api-repo`
- [ ] GitHub Actions service account created
- [ ] Service account has required IAM roles:
  - [ ] `roles/run.admin`
  - [ ] `roles/artifactregistry.writer`
  - [ ] `roles/iam.serviceAccountUser`
- [ ] Service account JSON key downloaded

### GitHub Repository Configuration
- [ ] Repository created/forked
- [ ] GitHub Secrets configured:
  - [ ] `GCP_SA_KEY` - Service account JSON key
  - [ ] `GCP_PROJECT_ID` - Your GCP project ID
  - [ ] `REGION` - Deployment region (e.g., us-central1)
  - [ ] `SERVICE_NAME` - Cloud Run service name
  - [ ] `POSTGRES_URL` - Supabase PostgreSQL connection string
  - [ ] `GOOGLE_API_KEY` - Google Gemini API key
  - [ ] `OPENAI_API_KEY` - OpenAI API key

### Database Setup
- [ ] Supabase project created
- [ ] PostgreSQL database accessible
- [ ] `message_store` table created
- [ ] Connection string tested

### API Keys
- [ ] Google Gemini API key obtained and tested
- [ ] OpenAI API key obtained and tested
- [ ] Both keys have sufficient quotas

## 🚀 Deployment Process

### Automatic Deployment (via GitHub Actions)
- [ ] Push code to `main` branch
- [ ] GitHub Actions workflow triggers automatically
- [ ] Monitor workflow in Actions tab
- [ ] Verify successful build
- [ ] Verify successful push to Artifact Registry
- [ ] Verify successful deployment to Cloud Run
- [ ] Note the deployed service URL

### Manual Deployment
- [ ] Authenticate with GCP: `gcloud auth login`
- [ ] Set project: `gcloud config set project PROJECT_ID`
- [ ] Build Docker image locally
- [ ] Push to Artifact Registry
- [ ] Deploy to Cloud Run with environment variables
- [ ] Verify deployment

## ✅ Post-Deployment Verification

### API Health Checks
- [ ] Access service URL: `https://[SERVICE_NAME]-[HASH]-[REGION].a.run.app`
- [ ] Check API docs: `/docs`
- [ ] Test `/generate` endpoint with sample request
- [ ] Test `/history/{user_id}` endpoint
- [ ] Verify conversation memory works across requests

### Monitoring
- [ ] Check Cloud Run metrics in GCP Console
- [ ] Review application logs
- [ ] Set up log-based alerts (optional)
- [ ] Configure uptime monitoring (optional)

### Configuration Validation
- [ ] Environment variables properly set
- [ ] Database connection working
- [ ] LLM providers responding
- [ ] Default LLM provider configured correctly
- [ ] History limits applying correctly
- [ ] Background tasks executing

## 🔧 Configuration Files

### Updated Files:
- [x] `Dockerfile` - Optimized for Cloud Run
- [x] `.github/workflows/docker-image.yml` - Complete CI/CD pipeline
- [x] `requirements.txt` - Pinned versions
- [x] `.dockerignore` - Build optimization
- [x] `DEPLOYMENT.md` - Complete deployment guide

### Configuration Review:
- [ ] `llm_config.json` - Review all settings
- [ ] `main.py` - Verify environment variable usage
- [ ] `.env` - Ensure not committed to Git

## 🔐 Security Checklist

- [ ] `.env` file in `.gitignore`
- [ ] Secrets stored in GitHub Secrets (not in code)
- [ ] Service account has minimal required permissions
- [ ] Container runs as non-root user
- [ ] Dependencies have no critical vulnerabilities
- [ ] HTTPS enforced (automatic with Cloud Run)
- [ ] Consider authentication for production:
  - [ ] Remove `--allow-unauthenticated` if needed
  - [ ] Implement API key authentication
  - [ ] Set up Cloud IAM for access control

## 📊 Resource Configuration

Current settings:
- **CPU**: 1 vCPU
- **Memory**: 512 Mi
- **Min Instances**: 0 (scales to zero)
- **Max Instances**: 10
- **Timeout**: 900s (15 minutes)
- **Concurrency**: 80 requests/instance

Adjust if needed for:
- [ ] Higher traffic
- [ ] Faster cold starts (increase min instances)
- [ ] More memory-intensive operations

## 🐛 Troubleshooting

### Common Issues and Solutions:

**Build fails in GitHub Actions:**
- [ ] Check all GitHub Secrets are set
- [ ] Verify Artifact Registry exists
- [ ] Review build logs for errors

**Deployment fails:**
- [ ] Verify service account permissions
- [ ] Check environment variables format
- [ ] Review Cloud Run logs

**API errors after deployment:**
- [ ] Test database connection
- [ ] Verify API keys are valid
- [ ] Check application logs
- [ ] Test locally with same configuration

**Slow response times:**
- [ ] Review history limits configuration
- [ ] Check database query performance
- [ ] Consider increasing memory/CPU
- [ ] Set min instances > 0

## 💰 Cost Monitoring

- [ ] Set up billing alerts
- [ ] Monitor Cloud Run usage
- [ ] Review request counts
- [ ] Optimize scaling settings
- [ ] Consider reserved instances for consistent traffic

## 📝 Rollback Plan

If deployment fails or issues arise:
```bash
# Rollback to previous revision
gcloud run services update-traffic SERVICE_NAME \
  --to-revisions=PREVIOUS_REVISION=100 \
  --region REGION

# List available revisions
gcloud run revisions list \
  --service=SERVICE_NAME \
  --region=REGION
```

## 🎯 Next Steps

After successful deployment:
- [ ] Document service URL
- [ ] Update API documentation
- [ ] Set up monitoring dashboards
- [ ] Configure automated backups
- [ ] Plan for scaling strategy
- [ ] Review and optimize costs
- [ ] Set up staging environment
- [ ] Implement CI/CD for staging
- [ ] Add integration tests
- [ ] Set up error monitoring (e.g., Sentry)

## 📞 Support Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Supabase Documentation](https://supabase.com/docs)

---

**Last Updated**: March 4, 2026
**Version**: 1.0
