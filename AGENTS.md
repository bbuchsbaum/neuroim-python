# AGENTS.md - Agent Workflow Instructions

## Issue Tracking with Beads

This project uses **beads_rust** (`br`/`bd`) for git-backed issue tracking. See https://github.com/Dicklesworthstone/beads_rust

### Essential Commands

| Command | Purpose |
|---------|---------|
| `bd ready` | List tasks without blockers (your next work) |
| `bd create --title="..." --priority=1` | Create task (P0=critical, P1=high, P2=medium, P3=low) |
| `bd show <id>` | View issue details and history |
| `bd update <id> --status=in_progress` | Mark task as in progress |
| `bd close <id> --reason="text"` | Close completed task |
| `bd dep add <child> <parent>` | Add dependency |
| `bd list --json` | List all open issues as JSON |
| `bd sync --flush-only` | Export DB to JSONL for git |

### Critical Rules for Agents

1. **NEVER use `bd edit`** - it opens an interactive editor. Use flag-based updates:
   ```bash
   bd update <id> --description="new description"
   bd update <id> --title="new title"
   ```

2. **Always use `--json` flag** for programmatic access

3. **Run `bd sync --flush-only` after changes** to ensure JSONL export for git

### Landing the Plane Protocol

When ending a work session, you MUST complete these steps in order:

1. **File remaining work** as new issues for anything not completed
2. **Run quality gates** (tests, linting, builds as appropriate)
3. **Update issue statuses** - close completed, update in-progress
4. **Sync and push**:
   ```bash
   bd sync --flush-only
   git add .beads/
   git commit -m "beads: sync issue state"
   git pull --rebase
   git push
   ```
5. **Verify clean state**: `git status` shows nothing pending
6. **Provide handoff context** for next session

**Work is NOT complete until `git push` succeeds.**

### Finding Work

```bash
bd ready --json          # Tasks without blockers
bd list --status=open    # All open tasks
bd stale --days 7        # Neglected tasks
bd search "keyword"      # Full-text search
```

### Key Concepts

- **Dependencies**: Issues can block other issues. `bd ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers 0-4)
- **Types**: task, bug, feature, epic, chore, docs, question
- **Blocking**: `bd dep add <issue> <depends-on>` to add dependencies
