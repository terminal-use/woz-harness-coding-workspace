# coding-workspace - Claude Agent SDK Template

This template creates an intelligent agent powered by the Claude Agent SDK. Your agent can have multi-turn conversations, read/write files, run commands, and maintain context across interactions.

## Features

- **Claude-Powered**: Uses Claude to understand and respond to user requests
- **File Operations**: Can create, read, and edit files in the workspace
- **Multi-Turn Conversations**: Maintains session context across message turns
- **Tool Access**: Has access to Read, Write, Edit, Bash, Glob, and Grep tools

## Key Concepts

- **Tasks**: A conversation thread or session with the agent
- **Messages**: User inputs and agent responses within a task
- **Session State**: Preserves conversation context across turns using session IDs

## Running the Agent

1. Run the agent locally:
```bash
terminaluse agents run --config config.yaml
```

The agent will start on port 8000 and print messages whenever it receives any of the ACP events.

## Local SDK Test Loop

Use these commands to test `coding_workspace` against local `sdks/python` changes with a deploy+smoke flow:

```bash
# from repo root
task coding-workspace:deploy
task coding-workspace:smoke PROJECT_ID=<your-project-id>
```

You can also run the smoke script directly:

```bash
cd apps/coding-envs/coding_workspace
PROJECT_ID=<your-project-id> ./scripts/smoke_local_sdk.sh
```

Optional env vars:
- `DEPLOY_FIRST=0` skips deploy and runs only smoke verification.
- `REPO_URL=<repo-url>` overrides the default smoke clone repo.
- `POLL_SECONDS=<seconds>` changes filesystem sync timeout.

## What's Inside

This template:
- Sets up an ACP server that routes messages to Claude
- Uses the Claude Agent SDK for intelligent conversation handling
- Maintains session state for multi-turn conversations
- Provides file and command execution capabilities out of the box

## Next Steps

For more advanced agent development, check out the TerminalUse tutorials:

- **Tutorials 00-08**: Learn about building synchronous agents with ACP
- **Tutorials 09-10**: Learn how to use Temporal to power asynchronous agents
  - Tutorial 09: Basic Temporal workflow setup
  - Tutorial 10: Advanced Temporal patterns and best practices

These tutorials will help you understand:
- How to handle long-running tasks
- Implementing state machines
- Managing complex workflows
- Best practices for async agent development

## The Config File

The `config.yaml` file is your agent's configuration file. It defines:
- How your agent should be built and packaged
- What files are included in your agent's Docker image
- Your agent's name and description
- Local development settings (like the port your agent runs on)

This file is essential for both local development and deployment of your agent.

## Project Structure

```
coding_workspace/
├── src/                      # Your agent's code
│   ├── __init__.py
│   └── agent.py              # Agent server and event handlers
├── Dockerfile               # Container definition
├── config.yaml            # Deployment config
├── dev.ipynb                # Development notebook for testing

└── pyproject.toml          # Dependencies (uv)

```

## Development

### 1. Customize Event Handlers
- Modify the handlers in `agent.py` to implement your agent's logic
- Add your own tools and capabilities
- Implement custom state management

### 2. Test Your Agent with the Development Notebook
Use the included `dev.ipynb` Jupyter notebook to test your agent interactively:

```bash
# Start Jupyter notebook (make sure you have jupyter installed)
jupyter notebook dev.ipynb

# Or use VS Code to open the notebook directly
code dev.ipynb
```

The notebook includes:
- **Setup**: Connect to your local TerminalUse backend
- **Task creation**: Create a new task for the conversation
- **Event sending**: Send events to the agent and get responses
- **Async message subscription**: Subscribe to server-side events to receive agent responses
- **Rich message display**: Beautiful formatting with timestamps and author information

The notebook automatically uses your agent name (`test/coding-workspace`) and demonstrates the async ACP workflow: create task → send event → subscribe to responses.

### 3. Manage Dependencies


You chose **uv** for package management. Here's how to work with dependencies:

```bash
# Add new dependencies
terminaluse uv add requests openai anthropic

# Install/sync dependencies
terminaluse uv sync

# Run commands with uv
uv run terminaluse agents run --config config.yaml
```

**Benefits of uv:**
- Faster dependency resolution and installation
- Better dependency isolation
- Modern Python packaging standards



### 4. Configure Credentials
Options:
1. Add any required credentials to your config.yaml via the `env` section
2. Export them in your shell: `export OPENAI_API_KEY=...`
3. For local development, create a `.env.local` file in the project directory

```python
import os
from dotenv import load_dotenv

if os.environ.get("ENVIRONMENT") == "development":
    load_dotenv()
```

## Local Development


### 1. Start the TerminalUse Backend
```bash
# Navigate to the backend directory
cd terminaluse

# Start all services using Docker Compose
make dev

# Optional: In a separate terminal, use lazydocker for a better UI (everything should say "healthy")
lzd
```

### 2. Setup Your Agent's requirements/pyproject.toml
```bash
terminaluse uv sync [--group editable-apy]
source .venv/bin/activate

# OR
conda create -n coding_workspace python=3.12
conda activate coding_workspace
pip install -r requirements.txt
```
### 3. Run Your Agent
```bash
# From this directory
export ENVIRONMENT=development && [uv run] terminaluse agents run --config config.yaml
```

### 4. Interact with Your Agent

Option 0: CLI (deprecated - to be replaced once a new CLI is implemented - please use the web UI for now!)
```bash
# Submit a task via CLI
terminaluse tasks submit --agent test/coding-workspace --task "Your task here"
```

Option 1: Web UI
```bash
# Start the local web interface
cd terminaluse-web
make dev

# Then open http://localhost:3000 in your browser to chat with your agent
```

## Development Tips

### Environment Variables
- Set environment variables in src/.env for any required credentials
- Or configure them in the config.yaml under the `env` section
- The `.env` file is automatically loaded in development mode

### To build the agent Docker image locally (normally not necessary):

1. Build the agent image:
```bash
terminaluse agents build --config config.yaml
```
