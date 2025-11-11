# 🧪 API Testing Guide - Swagger UI Requests

## 📍 Access Swagger UI

**URL:** http://localhost:8000/docs

---

## 🎯 Test Endpoint: `/api/v1/classify`

### **POST /api/v1/classify**

---

## ✅ **Test 1: High Confidence Keyword Match**

**Expected:** Immediate response, no LLM call

```json
{
  "user_input": "add this to my cart",
  "session_id": "test-session-001",
  "user_id": 12345,
  "context": {
    "product_id": "PROD-123",
    "product_name": "iPhone 15 Pro"
  }
}
```

**Expected Response:**
```json
{
  "intent": "ADD_TO_CART",
  "action_code": "ADD_TO_CART",
  "confidence": 0.95,
  "entities": {...},
  "reasoning": "Keyword match",
  "triggered_llm": false,
  "metadata": {
    "source": "keyword",
    "session_id": "test-session-001"
  }
}
```

---

## 🔍 **Test 2: Search Product Intent**

```json
{
  "user_input": "show me iPhone 15 cases",
  "session_id": "test-session-002",
  "user_id": 12345
}
```

**Expected:** High confidence product search

---

## 📊 **Test 3: View Cart Intent**

```json
{
  "user_input": "what's in my shopping cart?",
  "session_id": "test-session-003",
  "user_id": 12345
}
```

**Expected:** Cart viewing intent

---

## 🗑️ **Test 4: Remove from Cart**

```json
{
  "user_input": "remove the blue headphones from cart",
  "session_id": "test-session-004",
  "user_id": 12345,
  "context": {
    "cart_items": ["item-123", "item-456"]
  }
}
```

---

## 💳 **Test 5: Checkout Intent**

```json
{
  "user_input": "I want to checkout now",
  "session_id": "test-session-005",
  "user_id": 12345
}
```

---

## 📦 **Test 6: Track Order**

```json
{
  "user_input": "where is my order #12345?",
  "session_id": "test-session-006",
  "user_id": 12345,
  "context": {
    "order_id": "ORD-12345"
  }
}
```

---

## ⭐ **Test 7: View Reviews**

```json
{
  "user_input": "show me reviews for this product",
  "session_id": "test-session-007",
  "user_id": 12345,
  "context": {
    "product_id": "PROD-789"
  }
}
```

---

## 🔄 **Test 8: Initiate Return**

```json
{
  "user_input": "I want to return my order",
  "session_id": "test-session-008",
  "user_id": 12345,
  "context": {
    "order_id": "ORD-99999"
  }
}
```

---

## ❓ **Test 9: Unclear Intent (Triggers LLM)**

**Expected:** Low confidence, LLM fallback triggered

```json
{
  "user_input": "I'm not sure what I want",
  "session_id": "test-session-009",
  "user_id": 12345
}
```

**Expected Response:**
```json
{
  "intent": "UNCLEAR",
  "action_code": "UNCLEAR",
  "confidence": 0.2,
  "triggered_llm": true,
  "metadata": {
    "source": "LLM",
    "reason": "low_confidence"
  }
}
```

---

## 🤔 **Test 10: Ambiguous Intent**

```json
{
  "user_input": "Apple",
  "session_id": "test-session-010",
  "user_id": 12345
}
```

**Expected:** Ambiguous result, needs clarification

---

## 🎨 **Test 11: Complex Natural Language**

**Expected:** Hybrid classification (keyword + embedding)

```json
{
  "user_input": "I'm looking for a gift for my mom, maybe something tech-related under $100",
  "session_id": "test-session-011",
  "user_id": 12345,
  "context": {
    "budget": 100,
    "category": "gifts"
  }
}
```

---

## 📱 **Test 12: Multiple Intents**

```json
{
  "user_input": "add this to cart and show me similar products",
  "session_id": "test-session-012",
  "user_id": 12345,
  "context": {
    "product_id": "PROD-555"
  }
}
```

**Expected:** Primary intent (ADD_TO_CART) with secondary suggestions

---

## 🔔 **Test 13: Subscribe to Notifications**

```json
{
  "user_input": "notify me when this is back in stock",
  "session_id": "test-session-013",
  "user_id": 12345,
  "context": {
    "product_id": "PROD-OUT-OF-STOCK"
  }
}
```

---

## 💬 **Test 14: General Help**

```json
{
  "user_input": "how do I use this website?",
  "session_id": "test-session-014",
  "user_id": 12345
}
```

---

## 🌐 **Test 15: Language/Formatting Edge Case**

```json
{
  "user_input": "ADD 2 CART!!!",
  "session_id": "test-session-015",
  "user_id": 12345
}
```

**Expected:** Still recognizes ADD_TO_CART (case-insensitive, punctuation-tolerant)

---

## 🚫 **Test 16: Empty Input**

```json
{
  "user_input": "",
  "session_id": "test-session-016",
  "user_id": 12345
}
```

**Expected:** Validation error or UNCLEAR intent

---

## 🔤 **Test 17: Special Characters**

```json
{
  "user_input": "@@@ search for #iPhone $$$ 15 %%%",
  "session_id": "test-session-017",
  "user_id": 12345
}
```

**Expected:** Extracts "iPhone 15", returns SEARCH_PRODUCT

---

## 🌍 **Test 18: Very Long Input**

```json
{
  "user_input": "I was thinking about buying a new phone because my current one is getting old and slow, maybe something with a good camera because I like taking photos, and it should have good battery life too, what do you have in stock that fits this description and is reasonably priced?",
  "session_id": "test-session-018",
  "user_id": 12345
}
```

**Expected:** SEARCH_PRODUCT with extracted entities (camera, battery, phone)

---

## 💰 **Test 19: Price Inquiry**

```json
{
  "user_input": "how much does the iPhone 15 cost?",
  "session_id": "test-session-019",
  "user_id": 12345,
  "context": {
    "product_id": "IPHONE-15"
  }
}
```

---

## 🎁 **Test 20: Apply Coupon**

```json
{
  "user_input": "apply discount code SAVE20",
  "session_id": "test-session-020",
  "user_id": 12345,
  "context": {
    "coupon_code": "SAVE20"
  }
}
```

---

## 📈 **Test Scenarios Summary**

| Test # | Scenario | Expected Behavior |
|--------|----------|-------------------|
| 1-8 | **Clear intents** | High confidence (>0.8), keyword match |
| 9-10 | **Unclear/Ambiguous** | Low confidence, may trigger LLM |
| 11-12 | **Complex queries** | Hybrid classification, entity extraction |
| 13-14 | **Specific actions** | Correct intent with context |
| 15-17 | **Edge cases** | Robust handling, normalization |
| 18-20 | **Natural language** | Entity extraction, semantic understanding |

---

## 🎯 **What to Verify**

### ✅ **For Each Response, Check:**

1. **`intent`** field is correct
2. **`confidence`** score is reasonable (0-1)
3. **`action_code`** matches the intent
4. **`entities`** are extracted correctly
5. **`triggered_llm`** is true/false as expected
6. **`metadata`** contains useful debug info
7. **Response time** is acceptable (<200ms for keywords, <2s for LLM)

### ✅ **Response Structure Should Have:**

```json
{
  "intent": "string",           // Intent name
  "action_code": "string",      // Action to take
  "confidence": 0.0-1.0,        // Confidence score
  "entities": {},               // Extracted entities
  "reasoning": "string",        // Why this intent?
  "triggered_llm": boolean,     // Was LLM called?
  "metadata": {                 // Debug info
    "source": "keyword|embedding|LLM",
    "session_id": "string",
    "variant": "A|B",
    "latency_ms": number
  }
}
```

---

## 🔍 **Advanced Testing**

### **A/B Testing Verification**

Send same request multiple times with different `user_id`:

```json
// Variant A (even user_id)
{"user_input": "find shoes", "user_id": 1000}

// Variant B (odd user_id)
{"user_input": "find shoes", "user_id": 1001}
```

Check `metadata.variant` in response.

---

### **Session Context Testing**

Send sequential requests with same `session_id`:

```json
// Request 1
{
  "user_input": "show me laptops",
  "session_id": "context-test-001",
  "user_id": 5000
}

// Request 2 (with context from request 1)
{
  "user_input": "add the first one to cart",
  "session_id": "context-test-001",
  "user_id": 5000
}
```

System should remember context from previous message.

---

### **Performance Testing**

Send same request multiple times and check `metadata.latency_ms`:

```json
{
  "user_input": "search for phone",
  "session_id": "perf-test-{{$randomInt}}",
  "user_id": {{$randomInt}}
}
```

- **Keyword matches:** <50ms expected
- **Embedding matches:** <200ms expected
- **LLM fallback:** 500-2000ms expected

---

## 📊 **Expected Response Times**

| Intent Source | Expected Latency | What's Happening |
|--------------|------------------|------------------|
| **Keyword match** | <50ms | Direct regex/keyword lookup |
| **Embedding match** | 50-200ms | Semantic similarity search |
| **Hybrid (keyword+embedding)** | 100-300ms | Combines both methods |
| **LLM fallback** | 500-2000ms | OpenAI API call |
| **Cached LLM** | <100ms | Redis cache hit |

---

## 🚨 **Error Cases to Test**

### **Missing Required Fields**

```json
{
  "user_input": "test"
  // Missing session_id and user_id
}
```

**Expected:** 422 Validation Error

---

### **Invalid Data Types**

```json
{
  "user_input": 12345,
  "session_id": "test",
  "user_id": "not-a-number"
}
```

**Expected:** 422 Validation Error

---

### **Very Large Context**

```json
{
  "user_input": "test",
  "session_id": "large-context",
  "user_id": 9999,
  "context": {
    "data": "x".repeat(10000)  // Very large context
  }
}
```

**Expected:** Should handle gracefully or return error

---

## 🎉 **Quick Copy-Paste Requests**

### **High Confidence Keyword:**
```json
{"user_input": "add to cart", "session_id": "test-001", "user_id": 1}
```

### **Search Product:**
```json
{"user_input": "find iPhone cases", "session_id": "test-002", "user_id": 2}
```

### **Track Order:**
```json
{"user_input": "where is my order?", "session_id": "test-003", "user_id": 3}
```

### **Unclear (Triggers LLM):**
```json
{"user_input": "hmm not sure", "session_id": "test-004", "user_id": 4}
```

### **Edge Case:**
```json
{"user_input": "ADD!!! CART @@@", "session_id": "test-005", "user_id": 5}
```

---

## 📝 **Testing Checklist**

- [ ] Test all 20+ scenarios above
- [ ] Verify response structure matches schema
- [ ] Check confidence scores are reasonable
- [ ] Confirm entities are extracted correctly
- [ ] Verify LLM triggers only when needed
- [ ] Test A/B variant assignment works
- [ ] Check session context is maintained
- [ ] Verify response times are acceptable
- [ ] Test error handling for invalid inputs
- [ ] Confirm metadata contains debug info

---

## 🎯 **Expected Success Criteria**

✅ **Keyword intents:** >95% accuracy
✅ **Embedding intents:** >85% accuracy  
✅ **Response time:** <200ms for 95% of requests
✅ **LLM fallback:** Triggers only for unclear intents (<10% of requests)
✅ **Error handling:** Graceful degradation, no 500 errors

---

**Start testing!** Copy-paste these requests into Swagger UI and verify the responses match expectations! 🚀

