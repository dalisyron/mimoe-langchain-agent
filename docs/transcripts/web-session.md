# Web UI session against the real server (D3)

> Later change (2026-09-25): tool calls now show as one muted line each ("Read notes.md",
> "Calculating failed, trying a different approach"), with the arguments and the result one click
> away, instead of the cards with status chips described below. The SSE frames and the server
> behaviour recorded here are unchanged.

Date: 2026-09-25. Engine: mimOE Studio 0.6.5 at `http://localhost:8083/mimik-ai/openai/v1`
(engine v3.22.8 developer edition, node MacBookPro.lan) with `qwen3-4b` loaded; thinking off, `max_tokens`
at the app default, no downloads. The 1.0.27 runtime on 8093 stayed empty and untouched.

Server: `cd /Users/nibom/mimoe-assignment && uv run mimoe-agent serve --port 8000` (started in the background,
log in the scratchpad `serve.log`). Browser: headless Chromium 148.0.7778.96 driven by playwright-core 1.60.0
(the cached `chromium-1223` build of a sibling project, so nothing was downloaded), viewport 1180x820, light
colour scheme (dark for one screenshot). Driver scripts: scratchpad `webdrive/lib.cjs` and `s1.cjs`..`s5.cjs`.

Every page ran an init script that tees each `/api/*` fetch into an in-page log: request body, HTTP status
and content type, and the raw SSE frames as the browser received them, each stamped with milliseconds since
page load. The event lines below are those frames, decoded; runs of `token`/`thinking` frames are folded into
one line (`token x59 (8626-10201 ms)` = 59 frames between 8.6 s and 10.2 s) with their concatenated text.
Health polls are omitted. Assertions ran against the DOM after every step; all passed on the final run.

## 1. Server start and curl checks

`serve.log` at startup:

```
INFO:     connecting to mimOE Studio
INFO:     warming up qwen3-4b (tool probe)
INFO:     qwen3-4b answered with a structured ping call in 1.5 s
INFO:     agent ready: qwen3-4b on http://localhost:8083/mimik-ai/openai/v1 (v3.22.8 (developer edition)), tools enabled
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

`curl -s http://127.0.0.1:8000/api/health`:

```
{"mimoe_reachable":true,"model":"qwen3-4b","tokens_per_second":34.23856416340369,"max_context":12000,"node":"MacBookPro.lan","engine_version":"v3.22.8 (developer edition)","generation":"0.6","workspace":"/Users/nibom/mimoe-assignment/workspace","approval":"manual","network":"off","mode":"tools","error":null}
```

`curl -s http://127.0.0.1:8000/api/models` (summarised): loaded `[qwen3-4b: max_context 12000, tokens_per_second 34.24,
supports_tools null]`; registry `qwen3-4b (2497281312 B, ready)`, `smollm2-360m (386404992 B)`,
`qwen3-4b-instruct-2507 (2497281120 B)`, `smollm3-3b (1915305792 B)`, `qwen3-8b (5027783488 B)`,
`qwen3.5-4b (2740937888 B)`, all ready; `current: "qwen3-4b"`.

`curl -si http://127.0.0.1:8000/` -> `HTTP/1.1 200 OK`, `content-type: text/html; charset=utf-8`, `content-length: 488`
(the committed `web/dist/index.html`); `GET /assets/index-M10M8PN2.js` -> 200, `text/javascript`, 388,994 bytes
(the bundle committed before this step; the rebuilt one is `index-k1rcMez2.js`, see section 6).

## 2. Scenario 1: badge, list_files, CSV approval, New conversation, Deny (`s1.cjs`)

Steps and what the page showed:

1. Open `http://127.0.0.1:8000/`. Badge after the first health poll: `qwen3-4b 34.2 tok/s Studio 0.6 tools
   approval: manual network: off /Users/nibom/mimoe-assignment/workspace`; the empty state lists the five demo
   prompts as buttons. (Screenshot 01.)
2. Click the demo prompt **What files are in this workspace?** -> a `list_files` tool card (args
   `{"path": ".", "max_depth": 4}`, chip `done`, result `102 chars` collapsed) followed by the markdown answer
   rendered as a list, then the stats line `2 model calls · 96 tokens · 10.1 s · qwen3-4b`. (Screenshot 02.)
3. Send the demo prompt **How many rows does sales.csv have, and what is the total revenue?** -> the model
   asked for `run_python`; the approval panel appeared with the code (`pd.read_csv('sales.csv')`, `len(df)`,
   `df['revenue'].sum()`), the tool card switched to `needs approval`, the composer was disabled
   ("Waiting for the agent..."). (Screenshot 03.)
4. Click **Approve** -> the card became `done` with the result `exit_code: 0 / stdout: (8, np.float64(1836.6000000000001))`,
   the answer streamed: *The sales.csv file has 8 rows, and the total revenue is $1836.60.* (Screenshot 04.)
5. Click **New conversation** -> the transcript cleared back to the empty state; the next request used a new
   `thread_id` (`9ae7793d-...` -> `1865f187-...`). (Screenshot 05.)
6. Send **Use run_python to count the data rows of sales.csv and sum its revenue column.** -> approval panel;
   click **Deny** -> the card is `denied` with the rejection result expanded (*User rejected the tool call for
   `run_python` with reason: The user declined to run this code. ...*), the answer: *The code was not executed as
   the user declined to run it. ...*; the composer is enabled again. (Screenshot 06.)

Events as received by the page:

```
GET /api/models (99 ms)
GET /api/models (114 ms)
  -> 200 application/json (115 ms)
  end (121 ms) 
  -> 200 application/json (125 ms)
  end (125 ms) 
POST /api/chat (205 ms) {"thread_id":"9ae7793d-225e-4830-992a-08764bafa151","message":"What files are in this workspace?"}
  -> 200 text/event-stream; charset=utf-8 (264 ms)
  tool_call (4767 ms): {"id":"tool_0","name":"list_files","args":{"path":".","max_depth":4}}
  tool_result (4773 ms): {"id":"tool_0","name":"list_files","content":"README.md  (321 B)\nnotes.md  (307 B)\nsales.csv  (287 B)\nsrc/\nsrc/app.py  (916 B)\nsrc/utils.py  (109 B)","is_error":false}
  token x59 (8626-10201 ms): "The files in the workspace are:\n\n- `README.md` (321 B)\n- `notes.md` (307 B)\n- `sales.csv` (287 B)\n- `src/app.py` (916 B)\n- `src/utils.py` ("
  token x5 (10226-10326 ms): "109 B)"
  done (10356 ms): {"status":"completed","elapsed_s":10.148,"model":"qwen3-4b","usage_total":{"input_tokens":2402,"output_tokens":96,"llm_calls":2}}
  end (10358 ms) 
POST /api/chat (10431 ms) {"thread_id":"9ae7793d-225e-4830-992a-08764bafa151","message":"How many rows does sales.csv have, and what is the total revenue?"}
  -> 200 text/event-stream; charset=utf-8 (10435 ms)
  tool_call (16916 ms): {"id":"tool_0","name":"run_python","args":{"code":"import pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of rows\ntotal_rows = len(df)\n\n# Calculate the total revenue\ntotal_revenue = df['revenue'].sum()\n\ntotal_rows, total_revenue"}}
  approval_required (16916 ms): {"interrupt_id":"8c11ea7224f2a52c97c4833d1afa1b1b","action_requests":[{"name":"run_python","args":{"code":"import pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of rows\ntotal_rows = len(df)\n\n# Calculate the total revenue\ntotal_revenue = df['revenue'].sum()\n\ntotal_rows, total_revenue"},"description":"run_python wants to run this code:\n\n```python\nimport pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of rows\ntotal_rows = len(df)\n\n# Calculate the total revenue\ntotal_revenue = df['revenue'].sum()\n\ntotal_rows, total_revenue\n```\n\nthis code runs as you, with your files"}],"review_configs":[{"action_name":"run_python","allowed_decisions":["approve","reject"]}]}
  done (16918 ms): {"status":"awaiting_approval","elapsed_s":6.483,"model":"qwen3-4b","usage_total":{"input_tokens":1333,"output_tokens":90,"llm_calls":1}}
  end (16918 ms) 
POST /api/resume (17001 ms) {"thread_id":"9ae7793d-225e-4830-992a-08764bafa151","interrupt_id":"8c11ea7224f2a52c97c4833d1afa1b1b","decisions":["approve"]}
  -> 200 text/event-stream; charset=utf-8 (17010 ms)
  tool_result (17523 ms): {"id":"tool_0","name":"run_python","content":"exit_code: 0\nstdout:\n(8, np.float64(1836.6000000000001))\nstderr: (empty)","is_error":false}
  token x22 (22025-22621 ms): "The sales.csv file has 8 rows, and the total revenue is $1836.60."
  done (22651 ms): {"status":"completed","elapsed_s":5.648,"model":"qwen3-4b","usage_total":{"input_tokens":1472,"output_tokens":27,"llm_calls":1}}
  end (22651 ms) 
POST /api/chat (22953 ms) {"thread_id":"1865f187-7ecd-4060-82fd-6b8af532bbff","message":"Use run_python to count the data rows of sales.csv and sum its revenue column."}
  -> 200 text/event-stream; charset=utf-8 (22960 ms)
  tool_call (29043 ms): {"id":"tool_0","name":"run_python","args":{"code":"import pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of data rows\ndata_rows = len(df)\n\n# Sum the revenue column\nrevenue_sum = df['revenue'].sum()\n\n# Print the results\ndata_rows, revenue_sum"}}
  approval_required (29043 ms): {"interrupt_id":"1852e4ffd26457cdb924303cda10ab6a","action_requests":[{"name":"run_python","args":{"code":"import pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of data rows\ndata_rows = len(df)\n\n# Sum the revenue column\nrevenue_sum = df['revenue'].sum()\n\n# Print the results\ndata_rows, revenue_sum"},"description":"run_python wants to run this code:\n\n```python\nimport pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of data rows\ndata_rows = len(df)\n\n# Sum the revenue column\nrevenue_sum = df['revenue'].sum()\n\n# Print the results\ndata_rows, revenue_sum\n```\n\nthis code runs as you, with your files"}],"review_configs":[{"action_name":"run_python","allowed_decisions":["approve","reject"]}]}
  done (29043 ms): {"status":"awaiting_approval","elapsed_s":6.089,"model":"qwen3-4b","usage_total":{"input_tokens":1168,"output_tokens":92,"llm_calls":1}}
  end (29043 ms) 
POST /api/resume (29339 ms) {"thread_id":"1865f187-7ecd-4060-82fd-6b8af532bbff","interrupt_id":"1852e4ffd26457cdb924303cda10ab6a","decisions":["reject"]}
  -> 200 text/event-stream; charset=utf-8 (29343 ms)
  tool_result (29344 ms): {"id":"tool_0","name":"run_python","content":"User rejected the tool call for `run_python` with reason: The user declined to run this code. Tell the user it was not executed and stop; do not retry.","is_error":true}
  token x28 (33374-34111 ms): "The code was not executed as the user declined to run it. Please provide the necessary information or make adjustments to the code before attempting again."
  done (34140 ms): {"status":"completed","elapsed_s":4.798,"model":"qwen3-4b","usage_total":{"input_tokens":1304,"output_tokens":32,"llm_calls":1}}
  end (34140 ms) 
```

No page errors, no console errors.

## 3. Scenario 2: Stop mid-stream, follow-up turn, the two 409s (`s2.cjs`)

The UI never sends while a run or an approval is pending on its own thread (the composer is disabled), so the
server's per-thread 409s can only be reached from a second client on the same thread. Two tabs of one browser
context shared a thread id (`crypto.randomUUID` pinned to `e2e-shared-1790330432038` by an init script).

1. Tab A sends **Write a 300-word story about a lighthouse keeper. Do not use any tools, just write.**; after 15
   token frames the Stop button is visible.
2. Tab B sends **hi** on the same thread -> `POST /api/chat` 409; tab B shows an error block
   `HTTP 409: a run is in progress on thread e2e-shared-1790330432038` with the hint *Wait for the current turn to
   finish (or stop it), then send again.*; tab B's composer is enabled again. (Screenshot 09.)
3. Tab A clicks **Stop** after 27 token frames (133 characters of story) -> the fetch is aborted, the turn shows
   the partial text and the notice `Stopped.`, no stats line, the Send button is back and the textarea enabled.
   The tee saw the stream close 6 ms later. (Screenshot 07.)
4. Tab A sends **Say hello in three words.** on the same thread -> *Hello there!* in 3.9 s: the server released
   the thread lock on the disconnect. (Screenshot 08.)
5. Tab A sends the explicit run_python prompt -> approval panel. Tab B sends **hello again** -> 409; error block
   `HTTP 409: an approval is pending on this thread` with the hint *Approve or reject the pending run_python
   request first (POST /api/resume), or start a new conversation.* (Screenshot 10.)
6. Tab A clicks **Deny** so the thread is left clean.

Tab A events:

```
GET /api/models (98 ms)
GET /api/models (111 ms)
  -> 200 application/json (112 ms)
  end (118 ms) 
  -> 200 application/json (123 ms)
  end (123 ms) 
POST /api/chat (303 ms) {"thread_id":"e2e-shared-1790330432038","message":"Write a 300-word story about a lighthouse keeper. Do not use any tools, just write."}
  -> 200 text/event-stream; charset=utf-8 (365 ms)
  token x28 (4240-4930 ms): "Once upon a time, in a small coastal village, there lived a lighthouse keeper named Elias. Elias was known for his dedication and the"
  end (4936 ms) 
POST /api/chat (6520 ms) {"thread_id":"e2e-shared-1790330432038","message":"Say hello in three words."}
  -> 200 text/event-stream; charset=utf-8 (6528 ms)
  token x3 (10323-10378 ms): "Hello there!"
  done (10412 ms): {"status":"completed","elapsed_s":3.889,"model":"qwen3-4b","usage_total":{"input_tokens":1185,"output_tokens":7,"llm_calls":1}}
  end (10413 ms) 
POST /api/chat (10628 ms) {"thread_id":"e2e-shared-1790330432038","message":"Use run_python to count the data rows of sales.csv and sum its revenue column."}
  -> 200 text/event-stream; charset=utf-8 (10636 ms)
  tool_call (16810 ms): {"id":"tool_0","name":"run_python","args":{"code":"import pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of data rows\ndata_rows = len(df)\n\n# Sum the revenue column\nrevenue_sum = df['revenue'].sum()\n\n# Print the results\ndata_rows, revenue_sum"}}
  approval_required (16810 ms): {"interrupt_id":"99f79c8425b552820f423420a54f3848","action_requests":[{"name":"run_python","args":{"code":"import pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of data rows\ndata_rows = len(df)\n\n# Sum the revenue column\nrevenue_sum = df['revenue'].sum()\n\n# Print the results\ndata_rows, revenue_sum"},"description":"run_python wants to run this code:\n\n```python\nimport pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of data rows\ndata_rows = len(df)\n\n# Sum the revenue column\nrevenue_sum = df['revenue'].sum()\n\n# Print the results\ndata_rows, revenue_sum\n```\n\nthis code runs as you, with your files"}],"review_configs":[{"action_name":"run_python","allowed_decisions":["approve","reject"]}]}
  done (16813 ms): {"status":"awaiting_approval","elapsed_s":6.18,"model":"qwen3-4b","usage_total":{"input_tokens":1215,"output_tokens":90,"llm_calls":1}}
  end (16813 ms) 
POST /api/resume (17236 ms) {"thread_id":"e2e-shared-1790330432038","interrupt_id":"99f79c8425b552820f423420a54f3848","decisions":["reject"]}
  -> 200 text/event-stream; charset=utf-8 (17243 ms)
  tool_result (17243 ms): {"id":"tool_0","name":"run_python","content":"User rejected the tool call for `run_python` with reason: The user declined to run this code. Tell the user it was not executed and stop; do not retry.","is_error":true}
  token x22 (21476-22051 ms): "The code was not executed as the user declined to run it. Please provide further instructions or clarify your request."
  done (22078 ms): {"status":"completed","elapsed_s":4.839,"model":"qwen3-4b","usage_total":{"input_tokens":1351,"output_tokens":26,"llm_calls":1}}
  end (22078 ms) 
```

Tab B events (the `body:` lines are the JSON error bodies):

```
GET /api/models (35 ms)
GET /api/models (48 ms)
  -> 200 application/json (53 ms)
  end (54 ms) 
  -> 200 application/json (65 ms)
  end (65 ms) 
POST /api/chat (4462 ms) {"thread_id":"e2e-shared-1790330432038","message":"hi"}
  -> 409 application/json (4527 ms)
  body: {"detail":{"message":"a run is in progress on thread e2e-shared-1790330432038","hint":"Wait for the current turn to finish (or stop it), then send again."}}
  end (4527 ms) 
POST /api/chat (16825 ms) {"thread_id":"e2e-shared-1790330432038","message":"hello again"}
  -> 409 application/json (16832 ms)
  body: {"detail":{"message":"an approval is pending on this thread","hint":"Approve or reject the pending run_python request first (POST /api/resume), or start a new conversation."}}
  end (16832 ms) 
```

Console: two `Failed to load resource: ... 409 (Conflict)` lines, which Chromium logs for every non-2xx fetch;
no page errors.

## 4. Scenario 3: the model picker (`s3.cjs`)

1. The select lists *Loaded: qwen3-4b · 12000 ctx* and *Registry: smollm2-360m (0.4 GB), qwen3-4b-instruct-2507
   (2.5 GB), smollm3-3b (1.9 GB), qwen3-8b (5.0 GB), qwen3.5-4b (2.7 GB)*; **Load** is disabled while the current
   model is selected.
2. Select `smollm2-360m`, keep *unload current* checked, click **Load** -> status `loading smollm2-360m… a cold
   load can take a minute` with a spinner, composer disabled. `POST /api/model` answered 200 in 0.7 s; the status
   turned red: `smollm2-360m: smollm2-360m answered without a tool call in 0.1 s (finish_reason=stop, tool
   names=none, content='/no_think')`. The badge refreshed to `smollm2-360m 108.6 tok/s Studio 0.6 chat only ...`;
   the engine listed only `smollm2-360m`. (Screenshot 11.)
3. Send **What is the capital of France? Answer in one word.** -> *Paris*, no tool card,
   `1 model calls · 1 tokens · 0.1 s · smollm2-360m` (pluralisation fixed afterwards). (Screenshot 12.)
4. Select `qwen3-4b`, **Load** -> 200 in 2.1 s, status green: `qwen3-4b: qwen3-4b answered with a structured
   ping call in 1.1 s`; badge `qwen3-4b 33.4 tok/s Studio 0.6 tools ...`; engine lists only `qwen3-4b`.
   (Screenshot 13.)
5. Send **What files are in this workspace?** in the same conversation -> `list_files` card `done` and the
   answer: the rebuilt agent has its tools back and the thread survived the switches. (Screenshot 14.)

Server log for the two switches: `unloading qwen3-4b / loading smollm2-360m / connecting to mimOE Studio /
warming up smollm2-360m (tool probe) / WARNING: tool probe failed: ... chat-only mode ... / switched to
smollm2-360m (chat only)` and `unloading smollm2-360m / loading qwen3-4b / ... / qwen3-4b answered with a
structured ping call in 1.1 s / switched to qwen3-4b (tools)`. (D2 saw Studio 0.6.5 occasionally answer a load
with 201 without loading; the driver had a retry for that, it was not needed in four switches.)

Events:

```
GET /api/models (79 ms)
GET /api/models (93 ms)
  -> 200 application/json (94 ms)
  end (98 ms) 
  -> 200 application/json (105 ms)
  end (105 ms) 
POST /api/model (150 ms) {"model":"smollm2-360m","unload_previous":true}
  -> 200 application/json (814 ms)
  end (814 ms) 
GET /api/models (821 ms)
  -> 200 application/json (833 ms)
  end (834 ms) 
POST /api/chat (893 ms) {"thread_id":"20ade87b-c442-433f-9f68-954164596b7c","message":"What is the capital of France? Answer in one word."}
  -> 200 text/event-stream; charset=utf-8 (906 ms)
  token x1 (969-969 ms): "Paris"
  done (978 ms): {"status":"completed","elapsed_s":0.083,"model":"smollm2-360m","usage_total":{"input_tokens":114,"output_tokens":1,"llm_calls":1}}
  end (979 ms) 
POST /api/model (1201 ms) {"model":"qwen3-4b","unload_previous":true}
  -> 200 application/json (3336 ms)
  end (3337 ms) 
GET /api/models (3345 ms)
  -> 200 application/json (3359 ms)
  end (3359 ms) 
POST /api/chat (3402 ms) {"thread_id":"20ade87b-c442-433f-9f68-954164596b7c","message":"What files are in this workspace?"}
  -> 200 text/event-stream; charset=utf-8 (3408 ms)
  tool_call (7717 ms): {"id":"tool_0","name":"list_files","args":{"path":".","max_depth":4}}
  tool_result (7723 ms): {"id":"tool_0","name":"list_files","content":"README.md  (321 B)\nnotes.md  (307 B)\nsales.csv  (287 B)\nsrc/\nsrc/app.py  (916 B)\nsrc/utils.py  (109 B)","is_error":false}
  token x54 (11641-13070 ms): "The files in this workspace are:\n- README.md (321 B)\n- notes.md (307 B)\n- sales.csv (287 B)\n- src/app.py (916 B)\n- src/utils.py (109 B)"
  done (13100 ms): {"status":"completed","elapsed_s":9.697,"model":"qwen3-4b","usage_total":{"input_tokens":2448,"output_tokens":86,"llm_calls":2}}
  end (13101 ms) 
```

## 5. Rendering checks without inference (`s4.cjs`)

Dark colour scheme: badge `qwen3-4b 32.5 tok/s Studio 0.6 tools ...` on the dark palette (screenshot 15). At a
420 px wide viewport there is no horizontal overflow; the header wraps and the demo prompts wrap (screenshot 16,
which showed the bullet of a wrapped prompt sitting on its last line, fixed below).

## 6. Fixes made in `web/src` and the re-verification (`s5.cjs`)

Everything above worked against the real server, but the session exposed these UI issues, all fixed and covered
by vitest where the reducer is involved (27 tests pass, `tsc -b` clean, `npm run build` rebuilt `web/dist`):

- **Turn stats only showed the last run.** The server reports `done.usage_total` and `elapsed_s` per run (the
  chat that raised the approval, then the resume), so an approved turn ended with `1 model calls · 27 tokens ·
  5.6 s`. The reducer now sums the runs of a turn (`2 model calls · 122 tokens · 11.8 s`); the model shown is
  the one that spoke last. Test: *the stats of a turn add up its runs*.
- **Tool cards left "running…" for ever** when a run ended in an `error` event or an HTTP failure (for example
  a resume answered 409 because the interrupt went stale after a server restart). They now become `error` with
  the result *no result: the run ended before the tool reported back*, like Stop already did. Tests: *a resume
  that fails ... marks the approved tool*, *an error event after a tool_call ends the tool card too*.
- **`GET /api/models` was fetched twice on load** (the picker refetched when the badge's model went from
  unknown to `qwen3-4b`). It is fetched once the badge is known and again only when the current model changes.
- **Health polling during a model switch.** The server's `/api/health` answers `mimoe_reachable: false` with
  `error: "a model switch is in progress, please wait"` while a switch runs, which painted the dot red and a red
  hint line during any longer load. The badge no longer polls while the UI's own switch is in flight (the picker
  shows the progress) and refreshes once it ends.
- **Header layout**: with the picker and the button on the same row the badge wrapped into four lines; the
  header is now two rows (title + badge; picker + New conversation) and the workspace path gets more room.
- Pluralisation (`1 model call`, `1 token`) and the wrapped demo-prompt bullets (`vertical-align: top`).
- `web/.gitignore` ignores `.vitest/` (a JSON reporter directory written by the local vitest wrapper).

Re-verification on the rebuilt bundle (`index-k1rcMez2.js`): one `GET /api/models` at load; header 79 px high in
two rows (18 + 32 px); the explicit run_python prompt -> `1 model call · 92 tokens · 6.7 s` while pending, then
after **Approve** `2 model calls · 122 tokens · 11.8 s · qwen3-4b` and *The sales.csv file has 8 data rows, and
the sum of the revenue column is 1836.6.*; switch to `smollm2-360m` and back to `qwen3-4b` with the dot sampled
every 100 ms while the picker was busy: only `dot ok` seen, one health poll after each switch; engine ends with
`qwen3-4b` only. (Screenshots 17-19.)

```
GET /api/models (108 ms)
  -> 200 application/json (120 ms)
  end (125 ms) 
POST /api/chat (743 ms) {"thread_id":"a33c2caf-4dfa-44a2-9bd5-687edd51b170","message":"Use run_python to count the data rows of sales.csv and sum its revenue column."}
  -> 200 text/event-stream; charset=utf-8 (797 ms)
  tool_call (7414 ms): {"id":"tool_0","name":"run_python","args":{"code":"import pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of data rows\ndata_rows = len(df)\n\n# Sum the revenue column\nrevenue_sum = df['revenue'].sum()\n\n# Print the results\ndata_rows, revenue_sum"}}
  approval_required (7415 ms): {"interrupt_id":"a0b3df99c1851a3c51f91ad8c15ade7e","action_requests":[{"name":"run_python","args":{"code":"import pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of data rows\ndata_rows = len(df)\n\n# Sum the revenue column\nrevenue_sum = df['revenue'].sum()\n\n# Print the results\ndata_rows, revenue_sum"},"description":"run_python wants to run this code:\n\n```python\nimport pandas as pd\n\n# Load the CSV file\ndf = pd.read_csv('sales.csv')\n\n# Count the number of data rows\ndata_rows = len(df)\n\n# Sum the revenue column\nrevenue_sum = df['revenue'].sum()\n\n# Print the results\ndata_rows, revenue_sum\n```\n\nthis code runs as you, with your files"}],"review_configs":[{"action_name":"run_python","allowed_decisions":["approve","reject"]}]}
  done (7415 ms): {"status":"awaiting_approval","elapsed_s":6.667,"model":"qwen3-4b","usage_total":{"input_tokens":1168,"output_tokens":92,"llm_calls":1}}
  end (7415 ms) 
POST /api/resume (7678 ms) {"thread_id":"a33c2caf-4dfa-44a2-9bd5-687edd51b170","interrupt_id":"a0b3df99c1851a3c51f91ad8c15ade7e","decisions":["approve"]}
  -> 200 text/event-stream; charset=utf-8 (7686 ms)
  tool_result (8148 ms): {"id":"tool_0","name":"run_python","content":"exit_code: 0\nstdout:\n(8, np.float64(1836.6000000000001))\nstderr: (empty)","is_error":false}
  token x24 (12169-12801 ms): "The sales.csv file has 8 data rows, and the sum of the revenue column is 1836.6."
  done (12834 ms): {"status":"completed","elapsed_s":5.153,"model":"qwen3-4b","usage_total":{"input_tokens":1309,"output_tokens":30,"llm_calls":1}}
  end (12835 ms) 
POST /api/model (13078 ms) {"model":"smollm2-360m","unload_previous":true}
  -> 200 application/json (13725 ms)
  end (13725 ms) 
GET /api/models (13733 ms)
  -> 200 application/json (13744 ms)
  end (13744 ms) 
POST /api/model (13834 ms) {"model":"qwen3-4b","unload_previous":true}
  -> 200 application/json (15949 ms)
  end (15949 ms) 
GET /api/models (15958 ms)
  -> 200 application/json (15969 ms)
  end (15969 ms) 
```

## 7. Screenshots (scratchpad `webshots/`)

| file | what it shows |
|---|---|
| `01-empty-state.png` | first bundle: badge with model, tok/s, Studio 0.6, tools/approval/network chips, the picker, the five demo prompts |
| `02-list-files-answer.png` | `list_files` card (args, `done`, collapsed 102-char result), markdown list answer, stats line |
| `03-csv-approval-panel.png` | full page: the run_python card `needs approval`, the approval panel with the pandas code, Approve/Deny, composer disabled |
| `04-csv-approved-answer.png` | after Approve: card `done`, result collapsed (72 chars), answer with $1836.60 |
| `05-new-conversation.png` | the empty state again after New conversation |
| `06-deny-answer.png` | card `denied` with the rejection result expanded, the "not executed" answer, composer enabled |
| `07-stop-mid-stream.png` | tab A after Stop: partial story, `Stopped.` notice, Send button back |
| `08-stop-followup.png` | tab A: *Hello there!* on the same thread after the stop |
| `09-409-run-in-progress.png` | tab B: error block `HTTP 409: a run is in progress on thread ...` with the server hint |
| `10-409-approval-pending.png` | tab B: `HTTP 409: an approval is pending on this thread` with the hint |
| `11-switch-smollm2.png` | picker status (red) for the failed probe, badge `smollm2-360m ... chat only` |
| `12-chat-only-answer.png` | *Paris* without a tool card on smollm2-360m |
| `13-switch-back-qwen3-4b.png` | picker status (green) `answered with a structured ping call in 1.1 s`, badge back to `tools` |
| `14-after-switch-list-files.png` | list_files working again in the same conversation |
| `15-dark-mode.png` | the empty state in the dark colour scheme |
| `16-narrow-viewport.png` | 420 px wide, dark: header wrapped, no horizontal overflow (before the bullet fix) |
| `17-header-two-rows.png` | rebuilt bundle: the two-row header |
| `18-approved-stats.png` | rebuilt bundle: approved run_python turn with `2 model calls · 122 tokens · 11.8 s` |
| `19-after-switch-cycle.png` | rebuilt bundle: badge after the smollm2 -> qwen3-4b cycle |

## 8. Server-side observations (nothing blocked the UI)

- Event ordering matched CONTRACTS.md everywhere: `tool_call` before `approval_required`, then
  `done{status: awaiting_approval}`; the resume starts with the `tool_result` and ends with
  `done{status: completed}`; a rejection is a `tool_result` with `is_error: true`.
- Tool-call ids: with one call per run the shared alias table never had to rename anything; the resume's
  `tool_result` carried the chat run's `tool_0`, which is what the reducer keys on.
- `done.usage_total`/`elapsed_s` are per run, not per turn; the UI now sums them (see section 6). If the
  server ever reports thread totals instead, the reducer's `addStats` must stop adding.
- `/api/health` during a switch: `mimoe_reachable: false` plus the `error` text although the engine is fine;
  the UI skips polling during its own switches, but a switch started by another client (the CLI's `models use`)
  would still show a red dot for its duration.
- The rejected `tool_result` content includes the full REJECT_MESSAGE; the UI shows it verbatim in the card.
- Non-2xx fetches are logged by Chromium as console errors; that is browser behaviour, not the UI.

## 9. End state

The server was stopped with SIGTERM: `Shutting down / Waiting for application shutdown / Application shutdown
complete / Finished server process [81083]`; port 8000 is free. Over the whole session `serve.log` holds
12 `POST /api/chat` 200, 2 `POST /api/chat` 409, 5 `POST /api/resume` 200, 6 `POST /api/model` 200 and no
traceback. Engines afterwards: 8083 `GET /models` -> `['qwen3-4b']`; 8093 -> `[]`. Only `smollm2-360m` (0.4 GB)
was loaded transiently, three times, and unloaded again by the following switch back.
