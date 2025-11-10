
<img width="1000"  alt="image_" src="https://github.com/user-attachments/assets/0d966dde-f1d8-432f-bad0-aa79a5ccf396" />


### What this service is


This service is a smart proxy in front of llama.cpp that makes long‑context chat and IDE workflows much faster by managing llama.cpp slots, reusing cached context, and restoring saved caches from disk when needed. It speaks an OpenAI‑compatible Chat Completions API, so existing clients can connect without changes, including both streaming (SSE) and non‑stream responses depending on request settings.

### Why it’s needed

llama.cpp provides “slots,” each holding a conversation’s KV cache so repeated requests with the same or very similar prefix can skip recomputing the whole prompt and continue from the first mismatching token, which dramatically cuts latency for large prompts. In real teams the number of users can easily exceed the number of available slots (e.g., 20 developers but only 4 slots), so naive routing causes random slot reuse and cache overwrites that waste time and GPU/CPU cycles. This proxy solves that by steering requests to the right slot, saving evicted caches to disk, and restoring them on demand, so long prompts don’t need to be recomputed from scratch each time.

### How requests are balanced and slots are chosen

- Slots and heat: When a request lands in a slot and its cache is valid for reuse, the slot is considered “hot,” and new requests won’t overwrite it if other options exist, preserving useful KV for future reuse.
- Similarity matching: The proxy computes a fast, word‑block prefix similarity between the incoming conversation and existing hot slots, and only reuses a hot slot if the similarity meets a single ratio threshold (e.g., 85% of the shorter sequence), otherwise it rejects reuse to avoid polluting the hot cache with a weakly related prompt.
- Free and cold first: If reuse is rejected, the proxy sends the request to a free slot or a cold slot (one not currently carrying a valuable hot cache), protecting high‑value contexts from accidental overwrites under load.
- Oldest when full: If there are no free or cold slots, the proxy picks the least‑recently used slot and saves its current KV cache to disk before assigning the new request, ensuring nothing valuable is lost when the pool is exhausted.
- Restore on demand: When a new request matches a cache that was previously saved, the proxy restores that cache into a free/cold/oldest slot and routes the request there, which takes seconds versus minutes for full prompt recomputation on long contexts, especially in IDE scenarios with 30–60k tokens.
- Concurrency safety: Each slot is guarded with an async lock; if all are busy, the request waits for the first LRU slot to free, preventing race conditions and unintended cache overwrites during concurrent generation.


### Save and restore from disk

llama.cpp’s HTTP server exposes slot save/restore; saving writes a cache file to the directory provided by --slot‑save‑path, and restore loads by file basename (e.g., slotcache_<key>.bin), which is exactly how this proxy persists and revives caches across requests and restarts. The proxy keeps small local .meta files describing cached prefixes for fast lookup, while llama.cpp owns the actual KV .bin files under --slot‑save‑path for correctness and performance.

### Quick start

1) Start llama.cpp ( https://github.com/ggml-org/llama.cpp ) with slots and a cache directory:
```bash
llama-server -m ./model.gguf -np 4 --slot-save-path /var/kvcache --host 0.0.0.0 --port 8080 --swa-full
```

This enables the OpenAI‑compatible HTTP server, a pool of 4 slots, and a directory where slot KV caches are saved and restored by basename.

2) Run the proxy next to it:
```bash
git clone https://github.com/airnsk/proxycache.git
cd proxycache
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python3 proxycache.py  # or: uvicorn app:app --host 0.0.0.0 --port 8081
```

Your clients should call the proxy’s /v1/chat/completions endpoint; the proxy will handle similarity, slot selection, save/restore, and streaming vs non‑streaming automatically.


If you run into issues using gpt-oss-20b with an IDE like Cline, follow these instructions: https://www.reddit.com/r/CLine/comments/1mtcj2v/making_gptoss_20b_and_cline_work_together/

### Parameters

- LLAMA_SERVER_URL: The llama.cpp server base URL, e.g., http://127.0.0.1:8080, which must expose the OpenAI‑compatible chat completions endpoint.
- SLOTS_COUNT: The number of server slots (should match llama.cpp -np) so the proxy can track and plan reuse/restore correctly under load.
- SIMILARITY_MIN_RATIO: One similarity threshold (e.g., 0.85) controlling both active reuse and disk restore; if a match is below this ratio, the proxy will prefer a free/cold slot or restore instead of overwriting a hot slot.
- MIN_PREFIX_* (chars/words/blocks): Requests below this size are treated as “small” and steered to free/cold/oldest slots to avoid disturbing valuable hot caches used by large, long‑running prompts.
- LOCAL_META_DIR and --slot-save-path: The proxy stores small .meta descriptors locally for fast candidate lookup, while llama.cpp reads/writes the real KV cache files under --slot‑save‑path using basename in the HTTP API.


### Why this boosts IDE and long‑context productivity

For 30–60k‑token contexts typical in project‑wide IDE assistants, recomputing a full prompt can take minutes, whereas restoring a previously cached context and continuing from the first mismatching token typically takes seconds on llama.cpp, dramatically improving iteration speed for large teams with limited slots.
