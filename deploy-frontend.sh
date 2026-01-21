#!/bin/bash
# Deploy Streamlit frontend to Cloud Run and update API to point to it

set -e

REGION="europe-west1"
PROJECT_ID="mlops-license-plate-484109"
REPO="my-repo"
API_SERVICE="api"
FRONTEND_SERVICE="frontend"

API_URL="https://api-529952243062.europe-west1.run.app"

echo "🚀 Building frontend Docker image..."
docker buildx build --platform linux/amd64 \
  -f dockerfiles/frontend.dockerfile \
  -t europe-west1-docker.pkg.dev/${PROJECT_ID}/${REPO}/frontend:latest \
  --push \
  .

echo "📦 Deploying frontend to Cloud Run..."
FRONTEND_URL=$(gcloud run deploy ${FRONTEND_SERVICE} \
  --image europe-west1-docker.pkg.dev/${PROJECT_ID}/${REPO}/frontend:latest \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --set-env-vars BACKEND_URL=${API_URL} \
  --format="value(status.url)")

echo "✅ Frontend deployed at: ${FRONTEND_URL}"

echo "🔗 Updating API to point to frontend..."
gcloud run services update ${API_SERVICE} \
  --region ${REGION} \
  --set-env-vars STREAMLIT_URL=${FRONTEND_URL}

echo "✨ Done! Frontend: ${FRONTEND_URL}"
echo "   API: ${API_URL}"
