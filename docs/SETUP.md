# Brotato Setup Guide

Complete setup instructions for the Snap-to-Sheet Workout Tracker.

---

## Prerequisites

- Python 3.11+
- Google Cloud account
- Telegram account

---

## 1. Telegram Bot Setup

### Create Your Bot

1. Open Telegram and search for **@BotFather**
2. Send `/newbot`
3. Follow prompts to name your bot (e.g., "WorkoutLoggerBot")
4. **Save the Bot Token** - looks like: `7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxx`

**Your Bot Token**: `8564495560:AAF6XWawRAWYY7TfpYm_atQ72A2LJGbZow4`

### Set Webhook (after deploying)

```bash
curl -X POST "https://api.telegram.org/bot8564495560:AAF6XWawRAWYY7TfpYm_atQ72A2LJGbZow4/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://YOUR_DEPLOYMENT_URL/handle_telegram_webhook"}'
```

**Run this command after you deploy to get your webhook URL!**

---

## 2. Google Cloud Setup

### A. Create a Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or use existing: `projects/920262340940`

### B. Enable Required APIs

```bash
# Using gcloud CLI
gcloud services enable sheets.googleapis.com
gcloud services enable cloudfunctions.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

Or enable manually in Console:
- **Google Sheets API**: [Enable](https://console.cloud.google.com/apis/library/sheets.googleapis.com)
- **Cloud Functions API**: [Enable](https://console.cloud.google.com/apis/library/cloudfunctions.googleapis.com)

### C. Create Service Account for Sheets

1. Go to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)
2. Click **+ Create Service Account**
3. Name: `workout-sheets-writer`
4. Click **Create and Continue**
5. Grant role: **Editor** (or minimal: Sheets API access)
6. Click **Done**

### D. Generate Service Account Key

1. Click on your new service account
2. Go to **Keys** tab
3. Click **Add Key > Create new key**
4. Select **JSON**
5. Download and save securely

### E. Share Google Sheet with Service Account

1. Create a new Google Sheet for your workout log
2. Copy the **Spreadsheet ID** from URL:
   ```
   https://docs.google.com/spreadsheets/d/SPREADSHEET_ID_HERE/edit
   ```
3. Click **Share**
4. Add the service account email: `workout-sheets-writer@<project-id>.iam.gserviceaccount.com`
5. Grant **Editor** access

### F. Set Up Sheet Structure

Create a sheet named **"Workouts"** with headers in Row 1:

| B | C | D | E | F | G | H |
|---|---|---|---|---|---|---|
| Date | Exercise | Weight | Reps | Sets | Notes | Est. 1RM |

---

## 3. Gemini API Key

Your Gemini API key is already configured:

```
API Key: AIzaSyBzfbWb9xeQIpLhp78W61TFAxNhzWtdYkU
Project: projects/920262340940
```

> ⚠️ **Security Note**: Store this in environment variables, not in code!

---

## 4. Deploy Your Bot (FREE Options)

⚠️ **Google Cloud Functions requires billing**. Here are FREE alternatives:

### Option A: Railway.app (Recommended - FREE)

1. Go to [Railway.app](https://railway.app/) and sign up with GitHub
2. Click **New Project** > **Deploy from GitHub repo**
3. Select your bot repository
4. Add environment variables:
   - `TELEGRAM_BOT_TOKEN`: `8564495560:AAF6XWawRAWYY7TfpYm_atQ72A2LJGbZow4`
   - `GOOGLE_API_KEY`: `AIzaSyBzfbWb9xeQIpLhp78W61TFAxNhzWtdYkU`
   - `GOOGLE_SHEET_ID`: `1BeF2lNZEvwtjWyFzr70gBOrX9jVnNnsM0juQGAyqq4g`
   - `GOOGLE_CREDENTIALS`: Your full service account JSON (as string)
5. Railway will auto-deploy and give you a URL like: `https://your-app.railway.app`
6. Copy the URL and use it in the webhook curl command

### Option B: Render.com (FREE)

1. Go to [Render.com](https://render.com/) and sign up
2. Click **New +** > **Web Service**
3. Connect your GitHub repo
4. Configure:
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn main:app` (you'll need to create a Flask app wrapper)
5. Add environment variables (same as above)
6. Deploy and get your URL: `https://your-app.onrender.com`

### Option C: Google Cloud Functions (Requires Billing)

**Note**: Google requires billing info even for free tier usage.

Create a `.env.yaml` file (DO NOT commit to git):

```yaml
TELEGRAM_BOT_TOKEN: "8564495560:AAF6XWawRAWYY7TfpYm_atQ72A2LJGbZow4"
GOOGLE_API_KEY: "AIzaSyBzfbWb9xeQIpLhp78W61TFAxNhzWtdYkU"
GOOGLE_SHEET_ID: "1BeF2lNZEvwtjWyFzr70gBOrX9jVnNnsM0juQGAyqq4g"
GOOGLE_CREDENTIALS: '{"type":"service_account","project_id":"...full JSON key..."}'
```

Deploy Command:

```bash
gcloud functions deploy handle_telegram_webhook \
  --gen2 \
  --runtime=python311 \
  --region=us-central1 \
  --source=. \
  --entry-point=handle_telegram_webhook \
  --trigger-http \
  --allow-unauthenticated \
  --env-vars-file=.env.yaml
```

---

## 5. Test Locally

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Local Development Server

```bash
# Set environment variables
export TELEGRAM_BOT_TOKEN="8564495560:AAF6XWawRAWYY7TfpYm_atQ72A2LJGbZow4"
export GOOGLE_API_KEY="AIzaSyBzfbWb9xeQIpLhp78W61TFAxNhzWtdYkU"
export GOOGLE_SHEET_ID="1BeF2lNZEvwtjWyFzr70gBOrX9jVnNnsM0juQGAyqq4g"
export GOOGLE_CREDENTIALS='{"type":"service_account",...}'

# Run local server
functions-framework --target=handle_telegram_webhook --debug
```

---

## 6. Quick Checklist

- [ ] Create Telegram bot via @BotFather
- [ ] Save Telegram bot token
- [ ] Enable Google Sheets API
- [ ] Create service account
- [ ] Download service account JSON key
- [ ] Create Google Sheet with "Workouts" tab
- [ ] Share sheet with service account email
- [ ] Add headers to Row 1 (B:H)
- [ ] Deploy Cloud Function
- [ ] Set webhook URL in Telegram
- [ ] Send a workout screenshot to test!

---

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather | `7123456789:AAH...` |
| `GOOGLE_API_KEY` | Gemini API key | `AIzaSy...` |
| `GOOGLE_SHEET_ID` | Spreadsheet ID from URL | `1BxiMVs0XRA5nFMdKvBd...` |
| `GOOGLE_CREDENTIALS` | Full service account JSON (stringified) | `{"type":"service_account",...}` |

---

## Troubleshooting

### "No exercises found"
- Ensure screenshot is clear and readable
- Check Gemini API quota

### "Worksheet not found"
- Verify sheet is named exactly "Workouts"
- Check service account has access

### Webhook not working
- Verify webhook URL is correct
- Check Cloud Function logs: `gcloud functions logs read handle_telegram_webhook`
