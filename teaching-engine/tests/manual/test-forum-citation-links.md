# Manual E2E Test Plan: Forum Citation Links

## Prerequisites

1. Teaching engine server running on `http://localhost:8001`
   ```bash
   cd teaching-engine
   .venv/Scripts/python.exe -m uvicorn app.main:app --port 8001
   ```
2. Browser open to `http://localhost:8001`
3. ChromaDB populated with forum data (should include trumpetherald.com threads)

---

## Test 1: Forum citations appear in Knowledge mode

**Steps:**
1. Ensure the mode toggle is set to **Knowledge** (default)
2. Type a query likely to trigger forum results:
   > What do people on the forum say about Callet?
3. Wait for the bot to respond

**Expected results:**
- A "Community Discussion" box appears below the bot's message bubble
- The box contains one or more clickable links
- Each link displays the forum thread topic title as link text
- Links are blue/styled as hyperlinks

---

## Test 2: Forum links are clickable and open in new tab

**Steps:**
1. From Test 1, click on a forum discussion link in the "Community Discussion" box

**Expected results:**
- A new browser tab opens
- The URL points to `trumpetherald.com` (e.g., `https://www.trumpetherald.com/forum/viewtopic.php?t=XXXXX`)
- The URL does **not** contain a `sid` session parameter (e.g., no `&sid=abc123`)
- The forum thread loads (or shows a login page, which is acceptable)

---

## Test 3: Forum citations appear in Lesson mode

**Steps:**
1. Switch to **Lesson** mode using the mode toggle in the UI
2. Type a query likely to trigger forum results:
   > What do people on the forum say about Callet?
3. Wait for the bot to respond

**Expected results:**
- A "Community Discussion" box appears below the bot's message bubble
- Links are clickable and open in new tabs (same behavior as Knowledge mode)
- The link text shows the forum topic title

---

## Test 4: No forum box when only book results

**Steps:**
1. Switch to **Knowledge** mode
2. Type a query that should return only book results:
   > What does the Superchops book say about mouthpiece pressure?
3. Wait for the bot to respond

**Expected results:**
- Book citations appear (source badges like "SUPERCHOPS")
- No "Community Discussion" box appears below the message
- The forum-box div is absent from the DOM for this message

---

## Test 5: Verify API response in DevTools (Knowledge mode, streaming)

**Steps:**
1. Open browser DevTools (F12) and go to the **Network** tab
2. In **Knowledge** mode, send a query:
   > What do forum users think about tongue controlled embouchure?
3. In the Network tab, find the `/chat?stream=true` request
4. Click on it and view the **EventStream** or **Response** tab

**Expected results:**
- The last SSE event is the `done` event (contains `"done": true`)
- The done event JSON includes a `forum_citations` array
- Each entry in `forum_citations` has:
  - `url` field with a `trumpetherald.com` URL
  - `topic` field with the thread title
  - `tier` field set to `"forum"`
  - `era` field (e.g., `"GENERAL"`)

---

## Test 6: Verify API response in DevTools (Lesson mode, streaming)

**Steps:**
1. Open browser DevTools (F12) and go to the **Network** tab
2. Switch to **Lesson** mode
3. Send a query:
   > What do forum users say about practice routines?
4. In the Network tab, find the `/lesson?stream=true` request
5. View the EventStream/Response tab

**Expected results:**
- The done event includes a `forum_citations` array (same structure as Test 5)
- The done event also includes `lesson_state` and optionally `exercises`

---

## Test 7: Deduplication check

**Steps:**
1. In **Knowledge** mode, send a query that is likely to match multiple chunks from the same forum thread:
   > Tell me everything about Jerome Callet's embouchure method from the forum
2. Observe the "Community Discussion" box

**Expected results:**
- Each forum thread URL appears only once (no duplicate links)
- If multiple chunks came from the same thread, only one link is shown
- This can be verified in DevTools by checking the `forum_citations` array in the done event

---

## Test 8: Non-streaming fallback (optional)

**Steps:**
1. Using curl or Postman, send a non-streaming request:
   ```bash
   curl -X POST http://localhost:8001/chat \
     -H "Content-Type: application/json" \
     -d '{"text": "What do forum users say about Callet?"}'
   ```

**Expected results:**
- Response JSON includes `forum_citations` array with url and topic fields
- `citations` array contains book citations (if any book results were found)
- `forum_citations` and `citations` do not overlap
