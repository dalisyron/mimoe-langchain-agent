# Safety, honestly

The full version of the README's [safety section](../README.md#safety-honestly): what the agent does
to keep accidents small, and what gets past it.

The agent's own tools never touch the network. Code you approve runs as you, with your files and,
with `--allow-network`, your network. The approval prompt is the only boundary. `--auto-approve`
means trust-the-model. File contents are untrusted input.

The web server saves every conversation, including the file contents and tool output the agent
saw, in a SQLite file only your user can read (created 0600 in a 0700 folder on macOS and Linux).
Deleting a conversation in the sidebar removes its checkpoints too; `serve --history off` keeps
nothing on disk. The REPL never writes a history.

What `run_python` does to keep accidents small (none of it is a sandbox):

- The code runs in a fresh interpreter (`python -X utf8 -u -P _runner.py`) with the workspace as
  its working directory, stdin closed, and an allow-listed environment (`PATH`, `HOME`, `TEMP` and
  friends; none of your other variables, so no secrets leak into it). The workspace goes on
  `sys.path` only after start-up, so a `sitecustomize.py` in it cannot run before the approved
  code, and `import mimoe_agent` is blocked in the child.
- Limits, checked ten times a second: 30 s wall clock, more than 2 GB of resident memory across
  the snippet and everything it spawned (measured with `psutil`, because macOS enforces no memory
  rlimit and a runaway snippet would otherwise push the machine into swap), or 32 MB of combined
  output, and the whole process tree is killed (a process group `SIGKILL` on POSIX, `taskkill /T`
  on Windows). Files it writes are capped at 64 MB on POSIX (`RLIMIT_FSIZE`); the model sees at
  most 8 KB of output.
- Ctrl-C in the CLI and Stop in the web UI end the turn at once: the turn's cancel signal kills a
  running snippet's tree and closes the model's stream.
- What escapes those limits: the memory check is a poll, so a very fast allocation can overshoot
  for a fraction of a second before the kill; a snippet that calls `setsid` leaves the process
  group and survives the kill on POSIX; on Windows a grandchild whose parent already exited
  survives `taskkill /T` (no Job Objects); the socket guard (unless `--allow-network`,
  `socket.socket` raises "network disabled for run_python") does not block DNS lookups and is
  bypassable through the `_socket` module. It stops accidental network use, nothing more.
- Before you approve, both clients print a red warning when the code mentions sockets, `urllib`,
  `requests`, `httpx`, `subprocess`, `os.system`, `shutil.rmtree`, `os.remove` or opening a file
  for writing, and another when it contains invisible or terminal-control characters (zero-width
  and bidirectional-override characters, ESC sequences). Those are shown escaped, so code cannot
  look different on screen from what runs. Read the code anyway.

The other tools:

- `list_files`, `read_file`, `search_files` are jailed to the workspace: `..`, absolute paths,
  drive letters and UNC paths outside it, symlinks and junctions that point outside, and reserved
  Windows names are refused (an absolute path inside the workspace, which the model copies from the
  system prompt, counts as the relative path it names); listing and searching skip `.git`, `node_modules`, `.venv` and similar; credential
  files (`.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa*`, `credentials`, `.netrc`, `token.json` and
  similar, matched after Unicode normalisation and case folding) are refused, as are binaries and
  files over 2 MB. The jail is by path, so a workspace's `.git/config` is readable by `read_file`,
  and approved Python can of course read anything you can. A path that does not exist gets the
  closest workspace names in its error ("did you mean 'notes.md'? If that is the file the user
  means, read it."), never a credential file or anything behind a symlink: the model guesses
  names, and told only that its guess did not exist it gave up.
- `git` is read-only (`status`, `log`, `diff`) and runs without a shell, pager or prompt. It runs
  none of the repository's own code: hooks are pointed at the null device (`status` and `diff`
  rewrite the index, which fires `post-index-change`), filter drivers from the repository's config
  are disabled, and external diff, textconv and submodule recursion are off. Inherited `GIT_*`
  variables are dropped, so the tool always reports on the repository that contains the
  workspace; it times out after 20 s. The sample workspace lives in this repository, so `git log`
  shows my commits.
- Everything the model, a tool or the engine says is shown in the terminal with control
  sequences escaped: an ESC in a file or an answer could otherwise move the cursor, retitle the
  window or write your clipboard (OSC 52).
- `calculator` evaluates through an AST whitelist, no `eval`; expressions that would produce more
  than 100,000 digits are refused before they are computed.
- The web server binds `127.0.0.1` only, has no `--host` flag, accepts only `127.0.0.1`,
  `localhost` and `[::1]` as the `Host` header (a DNS-rebinding page in your browser cannot drive
  it), has no CORS and no authentication: it is a single-user tool. The UI renders model output as
  Markdown with raw HTML shown as text and `<img>` dropped, because an image URL in an answer would
  beacon file contents to any server through your browser.
- LangSmith tracing variables are forced to `false` unless you pass `--trace`, so a
  `LANGSMITH_TRACING=true` in your shell does not upload prompts. All HTTP clients ignore proxy
  variables.

The most dangerous combination prints a warning in the banner: `--auto-approve` with
`--allow-network` means "model-written code runs unattended and may open network connections; file
contents are untrusted input". A file in the workspace that tells the model to post your data
somewhere then has a path to do it.
