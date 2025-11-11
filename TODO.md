# TODO: Fix 403 Error for SEARCH_PRODUCT Intent

## Steps to Complete
- [x] Update app/config.py to add shopify_storefront_access_token setting
- [x] Modify app/llm_connectors/shopify_client.py to use storefront token for /api/ paths
- [x] Update app/orchestrator.py to initialize ShopifyClient with storefront token
- [x] Add token validation on startup
- [x] Build proper GraphQL query in orchestrator.py for SEARCH_PRODUCT based on entities
- [x] Test the fix with the provided API keys
