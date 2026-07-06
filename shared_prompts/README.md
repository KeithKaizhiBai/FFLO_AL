# Shared Prompts

This folder is the project-level prompt exchange area for Mac, iPhone, Windows
Codex, and future Codex sessions.

Use it to save long task prompts, reusable prompt templates, and session
handoff notes without pasting large files into remote chat.

## Directory Structure

```text
shared_prompts/
  inbox/      New prompts or task instructions added from any device.
  active/     The current prompt Codex should execute.
  archive/    Prompts that were completed, replaced, or abandoned.
  templates/  Reusable prompt templates.
  handoff/    Cross-session project status and next-step instructions.
  examples/   High-quality prompt examples worth preserving.
```

## Naming Rules

Use names like:

```text
YYYYMMDD_short_task_name.md
```

Use lowercase English letters, numbers, underscores, or short hyphens. Avoid
Chinese characters, spaces, and special symbols in filenames.

The canonical active prompt file is:

```text
shared_prompts/active/CURRENT_TASK_PROMPT.md
```

## Usage Flow

1. Put new prompt documents from Mac, iPhone, or another device into
   `shared_prompts/inbox/`.
2. Ask Codex to inspect `shared_prompts/inbox/` when you want to choose a new
   task.
3. Copy the selected prompt into
   `shared_prompts/active/CURRENT_TASK_PROMPT.md`.
4. Ask Codex to "read active prompt" or "continue from shared prompt".
5. After completion, copy or move the executed prompt into `archive/`.
6. After important work, update:
   - `shared_prompts/handoff/PROJECT_HANDOFF.md`
   - `shared_prompts/handoff/NEXT_ACTIONS.md`

## Large File Rule

Do not paste large files into Codex remote chat. Put prompt text in this folder
and keep large data, raw arrays, PDFs, images, and archives in their normal
project directories. Reference them by path from the prompt.

## Archive Rule

Do not overwrite archived prompts unless explicitly instructed. If a task is
rerun or revised, create a new timestamped prompt file.
