"""Coding Workspace Agent.

Clones a GitHub repo on task creation, configures GitHub auth, and assists
with planning, implementation, PR creation, and CI-fix loops.
"""

import os
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal, TypedDict

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import ResultMessage

from terminaluse.lib import (
    AgentServer,
    DataPart,
    TaskContext,
    make_logger,
)
from terminaluse.types import Event, TextPart as SDKTextPart

NOISY_RUNTIME_LOGGERS = (
    "httpx",
    "httpcore",
    "uvicorn.access",
    "terminaluse.lib.telemetry",
    "terminaluse.lib.sdk.fastacp",
    "opentelemetry",
    "opentelemetry.instrumentation",
)


GITHUB_HOST = "github.com"


AgentStatus = Literal["cloning", "warning", "error", "ready", "info"]


class AgentStatusData(TypedDict):
    kind: Literal["agent.status"]
    v: Literal[1]
    status: AgentStatus
    message: str


def _build_status_payload(status: AgentStatus, message: str) -> AgentStatusData:
    return {
        "kind": "agent.status",
        "v": 1,
        "status": status,
        "message": message,
    }


def _env_log_level(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip().upper()
    if not raw:
        return default
    return getattr(logging, raw, default)


def _configure_runtime_logging() -> None:
    """Reduce infra noise while keeping agent-level signal visible."""
    app_level = _env_log_level("AGENT_APP_LOG_LEVEL", logging.INFO)
    infra_level = _env_log_level("AGENT_INFRA_LOG_LEVEL", logging.WARNING)

    logging.getLogger().setLevel(app_level)
    for logger_name in NOISY_RUNTIME_LOGGERS:
        logging.getLogger(logger_name).setLevel(infra_level)


_configure_runtime_logging()
logger = make_logger(__name__)

WORKSPACE_DIR = "/workspace"
WORKSPACE_PATH = Path(WORKSPACE_DIR)
AGENT_SRC_DIR = Path(__file__).resolve().parent
BUNDLED_SKILLS_SOURCE_DIRS = (
    AGENT_SRC_DIR / "_embedded_skills",
    Path("/opt/coding-workspace-skills"),
    Path("/root/.claude/skills"),
    Path("/app/coding_workspace/skills"),
)
WORKSPACE_SKILLS_DIR = WORKSPACE_PATH / ".claude" / "skills"

SYSTEM_PROMPT = """You are a coding assistant working in a repository at /workspace.

Core behavior:
- Read the existing code before proposing edits.
- Be concise, practical, and explicit about assumptions.
- Prefer small, safe, testable changes.
- Explain what you changed and why.
- Use project skills from /workspace/.claude/skills when relevant.

When the user asks for a "plan":
- Provide a short numbered plan first.
- Update the plan as steps complete.

When asked to get changes "PR-ready":
1. Create a branch named `codex/<short-topic>`.
2. Implement requested changes.
3. Run relevant local checks (lint/tests/build).
4. Commit with a clear message.
5. Open or update a PR using `gh`.
6. Check CI status using `gh pr checks` / `gh run`.
7. If checks fail, inspect logs, fix, push, and repeat until green or blocked.
8. End with PR URL, check status summary, and any remaining risks.

GitHub tooling:
- If `gh` is available and authenticated, use it for PRs and CI checks.
- Never print or expose tokens in command output.
"""

server = AgentServer()


def _run(
    args: list[str],
    *,
    cwd: str | None = None,
    timeout: int = 120,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


async def _send_status(ctx: TaskContext, status: AgentStatus, message: str) -> None:
    payload = _build_status_payload(status, message)
    logger.info(
        "status_update task_id=%s status=%s message=%s",
        ctx.task.id,
        status,
        message,
    )
    await ctx.messages.send(
        DataPart(data=payload),
    )


def _configure_git_identity(github_login: str | None) -> None:
    name = github_login or "TerminalUse Agent"
    email = (
        f"{github_login}@users.noreply.github.com"
        if github_login
        else "terminaluse-agent@users.noreply.github.com"
    )
    _run(["git", "config", "user.name", name], cwd=WORKSPACE_DIR)
    _run(["git", "config", "user.email", email], cwd=WORKSPACE_DIR)


def _resolve_bundled_skills_dir() -> Path | None:
    for candidate in BUNDLED_SKILLS_SOURCE_DIRS:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _install_workspace_skills() -> tuple[bool, str | None]:
    """Copy bundled skills into the cloned repo so Claude can discover them."""
    bundled_skills_dir = _resolve_bundled_skills_dir()
    if bundled_skills_dir is None:
        return (
            False,
            "Bundled skills path missing: "
            + ", ".join(str(path) for path in BUNDLED_SKILLS_SOURCE_DIRS),
        )
    try:
        WORKSPACE_SKILLS_DIR.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            bundled_skills_dir,
            WORKSPACE_SKILLS_DIR,
            dirs_exist_ok=True,
        )
        return True, None
    except Exception as e:
        return False, str(e)


def _build_authenticated_clone_url(
    repo_url: str, github_token: str | None
) -> tuple[str, bool]:
    """Return a clone URL with embedded token when cloning from GitHub HTTPS."""
    if (
        not github_token
        or not repo_url.startswith("https://github.com/")
        or "@" in repo_url.split("://", 1)[1]
    ):
        return repo_url, False
    authed = repo_url.replace(
        "https://github.com/",
        f"https://x-access-token:{github_token}@github.com/",
        1,
    )
    return authed, True


def _redact_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "***")


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    # Never prompt in non-interactive runtime; produce deterministic errors.
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return env


def _task_param_str(ctx: TaskContext, key: str) -> str | None:
    params = getattr(ctx.task, "params", None)
    if not isinstance(params, dict):
        return None
    value = params.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _gh_auth_ready() -> bool | None:
    """Return True/False when gh is installed; None when gh is unavailable."""
    if shutil.which("gh") is None:
        return None
    status = _run(
        ["gh", "auth", "status", "--hostname", GITHUB_HOST],
        timeout=30,
        env=_git_env(),
    )
    return status.returncode == 0


async def _bootstrap_github_auth(
    ctx: TaskContext,
    *,
    github_token: str | None,
    github_login: str | None,
    repo_owner: str | None,
    repo_name: str | None,
) -> str:
    if not github_token:
        await _send_status(
            ctx,
            "warning",
            "No GitHub token available. PR creation and CI checks may be limited.",
        )
        return "missing_token"

    os.environ["GH_TOKEN"] = github_token
    os.environ["GITHUB_TOKEN"] = github_token

    if shutil.which("gh") is None:
        await _send_status(
            ctx,
            "warning",
            "GitHub token loaded, but `gh` CLI is not installed in this runtime.",
        )
        return "gh_missing"

    status = _run(["gh", "auth", "status", "--hostname", GITHUB_HOST])
    if status.returncode != 0:
        login = _run(
            ["gh", "auth", "login", "--hostname", GITHUB_HOST, "--with-token"],
            input_text=github_token,
        )
        if login.returncode != 0:
            await _send_status(
                ctx,
                "warning",
                "GitHub token detected, but `gh auth login` failed.",
            )
            return "gh_auth_failed"

    setup_git = _run(["gh", "auth", "setup-git"])
    if setup_git.returncode != 0:
        await _send_status(
            ctx,
            "warning",
            "`gh auth setup-git` failed; git pushes may require manual auth handling.",
        )
        return "gh_setup_git_failed"

    if repo_owner and repo_name:
        repo_view = _run(
            [
                "gh",
                "repo",
                "view",
                f"{repo_owner}/{repo_name}",
                "--json",
                "nameWithOwner",
            ],
        )
        if repo_view.returncode != 0:
            await _send_status(
                ctx,
                "warning",
                f"Authenticated to GitHub, but cannot access {repo_owner}/{repo_name}.",
            )
            return "gh_repo_access_failed"

    _configure_git_identity(github_login)
    return "ready"


@server.on_create
async def handle_create(ctx: TaskContext, params: dict[str, Any]):
    """Handle task creation - clone repo directly (SDK wrapper handles sync)."""

    repo_url = params.get("repo_url")
    repo_owner = params.get("repo_owner")
    repo_name = params.get("repo_name")
    github_token = params.get("github_token")
    github_login = params.get("github_login")
    logger.info(
        "task_create_received task_id=%s repo_url=%s repo_owner=%s repo_name=%s has_github_token=%s",
        ctx.task.id,
        repo_url,
        repo_owner,
        repo_name,
        bool(github_token),
    )

    # Initialize state
    await ctx.state.create(
        state={
            "session_id": None,
            "repo_url": repo_url,
            "github_auth_status": (
                "pending" if github_token else "missing_token"
            ),
        },
    )

    if not repo_url:
        await _send_status(ctx, "error", "Missing `repo_url` task param.")
        return

    # Send cloning status
    await _send_status(ctx, "cloning", f"Cloning {repo_url} ...")
    if not github_token and repo_url.startswith("https://github.com/"):
        await _send_status(
            ctx,
            "warning",
            "No GitHub token was provided. Private repositories may fail to clone.",
        )

    try:
        clone_url, used_embedded_token = _build_authenticated_clone_url(
            repo_url, github_token
        )
        logger.info(
            "clone_start task_id=%s clone_url_redacted=%s using_embedded_token=%s",
            ctx.task.id,
            "https://github.com/***" if clone_url.startswith("https://") else clone_url,
            used_embedded_token,
        )

        clone_cmd = ["git"]
        # Avoid helper prompts and stale credential helper issues for anonymous clones.
        if not github_token:
            clone_cmd.extend(["-c", "credential.helper="])
        clone_cmd.extend(
            ["clone", "--depth", "1", clone_url, WORKSPACE_DIR]
        )

        # Clone directly with subprocess
        result = _run(clone_cmd, timeout=300, env=_git_env())
        logger.info(
            "clone_finished task_id=%s return_code=%s",
            ctx.task.id,
            result.returncode,
        )

        if result.returncode != 0:
            stderr = (result.stderr or "").strip() or "Unknown git clone failure"
            stderr = _redact_secret(stderr, github_token)
            if (
                not github_token
                and "could not read Username for 'https://github.com'" in stderr
            ):
                stderr = (
                    "GitHub requested credentials. The repository may be private "
                    "or inaccessible without authentication. Reconnect GitHub and retry."
                )
            logger.warning(
                "clone_failed task_id=%s reason=%s",
                ctx.task.id,
                stderr,
            )
            await _send_status(ctx, "error", f"Clone failed: {stderr}")
            return

        # Ensure repository origin URL never stores an embedded token.
        if used_embedded_token:
            _run(
                ["git", "remote", "set-url", "origin", repo_url],
                cwd=WORKSPACE_DIR,
            )

        skills_ready, skills_error = _install_workspace_skills()
        if skills_ready:
            logger.info("workspace_skills_installed task_id=%s", ctx.task.id)
        else:
            logger.warning(
                "workspace_skills_install_failed task_id=%s error=%s",
                ctx.task.id,
                skills_error,
            )
            await _send_status(
                ctx,
                "warning",
                "Workspace is running, but bundled skills could not be installed.",
            )

        await _send_status(
            ctx,
            "ready",
            "Repository cloned successfully. Filesystem sync is handled automatically.",
        )
        github_auth_status = await _bootstrap_github_auth(
            ctx,
            github_token=github_token,
            github_login=github_login,
            repo_owner=repo_owner,
            repo_name=repo_name,
        )
        logger.info(
            "task_create_ready task_id=%s github_auth_status=%s",
            ctx.task.id,
            github_auth_status,
        )
        await ctx.state.update(
            {
                "github_auth_status": github_auth_status,
            }
        )
        if github_auth_status == "ready":
            await _send_status(
                ctx,
                "info",
                "GitHub authentication configured for PR + CI workflows.",
            )
        await ctx.messages.send(
            "Workspace is ready. Ask me to plan work, create a PR, and "
            "iterate on CI failures until checks pass."
        )

    except subprocess.TimeoutExpired:
        logger.warning("clone_timeout task_id=%s", ctx.task.id)
        await _send_status(ctx, "error", "Clone timed out.")
    except Exception as e:
        logger.exception("clone_error task_id=%s error=%s", ctx.task.id, e)
        await _send_status(ctx, "error", f"Clone error: {str(e)}")


@server.on_event
async def handle_event(ctx: TaskContext, event: Event):
    """Handle incoming messages from users."""
    try:
        if not isinstance(event.content, SDKTextPart):
            raise ValueError("Only text messages supported.")
        user_message = event.content.text
        logger.info(
            "task_event_received task_id=%s message_chars=%s",
            ctx.task.id,
            len(user_message),
        )

        state = await ctx.state.get()
        session_id = state.get("session_id") if state else None
        github_auth_status = (
            state.get("github_auth_status")
            if isinstance(state, dict)
            else None
        )
        task_github_token = _task_param_str(ctx, "github_token")
        task_github_login = _task_param_str(ctx, "github_login")
        task_repo_owner = _task_param_str(ctx, "repo_owner")
        task_repo_name = _task_param_str(ctx, "repo_name")

        if task_github_token:
            # Rehydrate auth env for every event in case runtime was rescheduled.
            os.environ["GH_TOKEN"] = task_github_token
            os.environ["GITHUB_TOKEN"] = task_github_token

        should_retry_github_auth = False
        if task_github_token:
            retryable_auth_statuses = {
                None,
                "pending",
                "missing_token",
                "gh_missing",
                "gh_auth_failed",
                "gh_setup_git_failed",
                "gh_repo_access_failed",
            }
            if github_auth_status in retryable_auth_statuses:
                should_retry_github_auth = True
            elif github_auth_status == "ready":
                gh_ready = _gh_auth_ready()
                should_retry_github_auth = gh_ready is False

        if should_retry_github_auth:
            logger.info(
                "task_event_rebootstrap_github task_id=%s has_task_token=%s previous_status=%s",
                ctx.task.id,
                bool(task_github_token),
                github_auth_status,
            )
            refreshed_github_auth_status = await _bootstrap_github_auth(
                ctx,
                github_token=task_github_token,
                github_login=task_github_login,
                repo_owner=task_repo_owner,
                repo_name=task_repo_name,
            )
            await ctx.state.update(
                {"github_auth_status": refreshed_github_auth_status}
            )

        if not WORKSPACE_SKILLS_DIR.exists():
            skills_ready, skills_error = _install_workspace_skills()
            if skills_ready:
                logger.info(
                    "workspace_skills_rehydrated task_id=%s",
                    ctx.task.id,
                )
            else:
                logger.warning(
                    "workspace_skills_rehydrate_failed task_id=%s error=%s",
                    ctx.task.id,
                    skills_error,
                )

        runtime_system_prompt = SYSTEM_PROMPT
        if task_github_token:
            runtime_system_prompt += (
                "\nRuntime facts:\n"
                "- GH_TOKEN and GITHUB_TOKEN are configured for this task.\n"
                "- Before claiming auth is missing, verify with "
                "`gh auth status --hostname github.com`."
            )
        else:
            runtime_system_prompt += (
                "\nRuntime facts:\n"
                "- No task-scoped GitHub token was provided."
            )

        options = ClaudeAgentOptions(
            include_partial_messages=True,
            permission_mode="bypassPermissions",
            cwd=WORKSPACE_DIR,
            allowed_tools=["Read", "Write", "Bash", "Edit", "Grep", "Glob"],
            setting_sources=["project", "user"],
            resume=session_id,
            system_prompt=runtime_system_prompt,
        )

        async for message in query(prompt=user_message, options=options):
            await ctx.messages.send(message)
            if isinstance(message, ResultMessage):
                await ctx.state.update({"session_id": message.session_id})
                logger.info(
                    "session_updated task_id=%s session_id=%s",
                    ctx.task.id,
                    message.session_id,
                )

    except Exception as e:
        logger.exception("task_event_error task_id=%s error=%s", ctx.task.id, e)
        await ctx.messages.send(f"Error: {str(e)}")


@server.on_cancel
async def handle_cancel(ctx: TaskContext):
    logger.info(f"Task cancelled: {ctx.task.id}")
