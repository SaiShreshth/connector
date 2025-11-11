# Orchestrator Code Analysis & Usage Guide

## Issues Found

### 1. **Shopify Domain Configuration Issue** ⚠️ CRITICAL
- **Problem**: Your `.env` has `SHOPIFY_SHOP_DOMAIN=happyruh.com`
- **Should be**: `SHOPIFY_SHOP_DOMAIN=happyruh.myshopify.com`
- **Evidence**: Logs show 301 redirects from `happyruh.com` → `happyruh.myshopify.com`
- **Fix**: Update your `.env` file

### 2. **Code Not Reloaded** ⚠️
- **Problem**: Old log messages appearing (line 60 shows old format)
- **Evidence**: New logging like "ORCHESTRATOR: Received..." not appearing
- **Fix**: Restart orchestrator completely (stop and start fresh)

### 3. **Orchestrator Usage** ✅
- **Status**: Code structure is correct
- **Endpoints**:
  - `POST /orchestrate` - Main orchestration endpoint
  - `GET /debug/shopify/ping` - Test Shopify connection
  - `GET /` - Health check

## How to Use the Orchestrator Correctly

### Step 1: Fix Environment Variables
Update your `.env` file:
```env
SHOPIFY_SHOP_DOMAIN=happyruh.myshopify.com  # NOT happyruh.com
SHOPIFY_ACCESS_TOKEN=your_token_here
PG_DSN=your_postgres_dsn
```

### Step 2: Start Orchestrator
```powershell
# Terminal 1: Start orchestrator on port 8001
uvicorn app.orchestrator:app --host 0.0.0.0 --port 8001 --reload
```

### Step 3: Test the Orchestrator

**Option A: Test via Debug Endpoint**
```powershell
curl http://localhost:8001/debug/shopify/ping
```

**Option B: Test Full Orchestration**
```powershell
curl -X POST http://localhost:8001/orchestrate `
  -H "Content-Type: application/json" `
  -d '{"text": "search for shoes"}'
```

**Option C: Use FastAPI Docs**
- Open: http://localhost:8001/docs
- Test `/orchestrate` endpoint with: `{"text": "search for shoes"}`

## Expected Log Flow

When you call `/orchestrate` with a search query, you should see:

1. `ORCHESTRATOR: Received /orchestrate request | text='...'`
2. `ORCHESTRATOR: Intent classification result | action_code=SEARCH_PRODUCT ...`
3. `ORCHESTRATOR: Parsed action_code=SEARCH_PRODUCT`
4. `Routing decision | action=SEARCH_PRODUCT ... shopify=True`
5. `ORCHESTRATOR: Calling Shopify via connector | path=...`
6. `Shopify request start | corr_id=... method=POST path=...`
7. `Preparing to log | corr_id=... status_code=200 ...`
8. `insert_shopify_call called | corr_id=... pool_exists=True`
9. `About to execute INSERT | corr_id=...`
10. `INSERT executed successfully | corr_id=...`
11. `Shopify call logged to Postgres | ...`
12. `ORCHESTRATOR: Shopify response | status=200 ...`

## Common Issues & Solutions

### Issue: No logs appearing
- **Solution**: Make sure you're calling port 8001 (orchestrator), not 8000 (main.py)
- **Check**: Look for "ORCHESTRATOR:" prefix in logs

### Issue: 301 redirects being logged
- **Solution**: Fix `.env` domain to `happyruh.myshopify.com`
- **Note**: With `follow_redirects=True`, redirects are followed automatically

### Issue: Database not updating
- **Check logs for**:
  - `insert_shopify_call called` - confirms function is called
  - `About to execute INSERT` - confirms SQL execution
  - `INSERT executed successfully` - confirms DB write
  - Any error messages after these

### Issue: Shopify client not configured
- **Check**: Startup logs should show "Shopify client initialized"
- **Fix**: Ensure `SHOPIFY_SHOP_DOMAIN` and `SHOPIFY_ACCESS_TOKEN` are set

## Code Flow Summary

```
User Request → /orchestrate
  ↓
Call Intent Classifier (port 8000)
  ↓
Parse Intent (e.g., SEARCH_PRODUCT)
  ↓
Route to Shopify URL (happyruh.myshopify.com)
  ↓
Use ShopifyClient.post() → Logs to Postgres
  ↓
Return Response
```

## Verification Checklist

- [ ] `.env` has correct `SHOPIFY_SHOP_DOMAIN=happyruh.myshopify.com`
- [ ] Orchestrator running on port 8001
- [ ] PostgreSQL pool initialized (check startup logs)
- [ ] Shopify client initialized (check startup logs)
- [ ] Making requests to `http://localhost:8001/orchestrate` (not 8000)
- [ ] Seeing "ORCHESTRATOR:" prefixed logs
- [ ] Database has entries in `shopify_api_calls` table

