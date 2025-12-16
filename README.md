# FIBO Brand Studio – JSON‑Native Agentic Workflow

<div align="center">

[![Watch the video](https://img.youtube.com/vi/IerwvpGiVmA/0.jpg)](https://youtu.be/IerwvpGiVmA)

</div>

An agentic FastAPI backend and Lovable/Vite frontend that turn brand briefs (plus optional reference images) into structured JSON, use that JSON to drive the FIBO model and LoRA color controls, and automatically check each image for on‑brand color palette and gaze before showing it to the user.

## Hackathon Category – Best JSON‑Native or Agentic Workflow

This project treats JSON as the single source of truth for every FIBO image:

- The agent converts loose creative requests and reference images into rich JSON prompts that capture brand colors, style, and composition.
- That JSON directly steers FIBO and a color‑palette LoRA, so every generation pass is traceable, repeatable, and easy to re‑run at scale.
- After each image is produced, JSON results from palette and gaze tools decide whether the asset is “approved” or needs another pass.

In practice, this feels less like “prompting a model” and more like plugging FIBO into a production‑grade pipeline: teams describe the campaign once, then the agent automatically generates, evaluates, and improves visuals while keeping a full JSON trail for every decision.

## What the project does

- Turns brand briefs and campaign asks into structured JSON prompts (with palettes, aesthetics, and layout hints) using Gemini.
- Uses those JSON prompts to steer FIBO via BRIA Vision Studio or a LoRA‑powered FIBO pipeline.
- Automatically checks the final image against the brand palette and desired gaze direction.
- Surfaces clear, human‑readable feedback (“missing brand yellow”, “no face detected”) alongside URLs for generated images and overlays.
- Exposes everything through a simple `/ask` endpoint plus a Lovable frontend for creative review.

## How the agentic workflow is wired

The core orchestration lives in `backend/src/agent.py` and `backend/main_fastapi.py`:

- **Chat entrypoint (`/ask`)** – `backend/main_fastapi.py:35` accepts a `chat_id`, `query`, optional `seed`, and optional base64 input image, then calls `run_agent` from `backend/src/agent.py`.
- **LangGraph agent state** – `State` tracks the conversation messages plus `remaining_steps`, `image_url`, `structured_prompt`, and `seed`, persisted via `SqliteSaver` so each `chat_id` behaves like a durable thread.
- **Tool‑driven workflow** – The agent uses LangChain tools from `backend/src/tools`:
  - `generate_image.py` – Calls BRIA Vision Studio’s FIBO backend or a LoRA‑enhanced FIBO pipeline. When it detects hex color codes in the prompt it:
    - Builds a JSON structured prompt with Gemini (`get_json_prompt` in `backend/src/fibo_lora_prompt_generation/prompt_to_json.py`).
    - Runs `generate_image_lora` from `backend/src/bria_lora_pipeline.py` to enforce the target palette.
    - Saves the output to a temp file and exposes it via `/download_temp_img`.
  - `generate_color_palette.py` – Uses OpenAI to propose a JSON color palette and renders it as an image for quick visual review.
  - `color_checker.py` – Analyzes an image’s dominant colors and compares them to required hex codes, returning success/miss information and coverage scores.
  - `gaze_tracker.py` – Sends the image to a local MCP server (`backend/gaze_mcp`) to estimate gaze directions and returns an annotated overlay image.
- **Single‑shot image generation** – A `ToolCallLimitMiddleware` ensures `generate_image` is only called once per agent run, which keeps the workflow predictable and cost‑aware.
- **JSON everywhere** – Structured prompts, LoRA palettes, compliance checks, and gaze results all move through the agent as JSON, making it easy to log, replay, and plug into other systems.

## Repository layout

- `backend/` – Python package, FastAPI app, and agentic workflow:
  - `main_fastapi.py` – FastAPI app with `/ask`, `/segment`, and `/download_temp_img` endpoints.
  - `entrypoint.sh` – Container entrypoint that starts the gaze MCP server and Uvicorn.
  - `src/agent.py` – LangGraph agent wiring (state, tools, checkpointer).
  - `src/tools/` – LangChain tools for image generation, palette suggestion, palette checking, and gaze tracking.
  - `src/bria.py` – BRIA Vision Studio client used to hit FIBO.
  - `src/bria_lora_pipeline.py` – LoRA‑enabled FIBO pipeline with post‑processing to snap colors back to the desired palette.
  - `src/fibo_lora_prompt_generation/` – Gemini‑based JSON promptifier that turns text and images into structured prompts.
  - `src/segmentor.py`, `src/image_utils.py` – SAM‑based segmentation and color extraction utilities.
  - `.env.dist` – Example backend environment file.
- `lovable-frontend/` – Vite + React experience for chatting with the agent, previewing images, and inspecting brand‑compliance overlays.
- `docker-compose.yml` – Orchestrates the frontend, backend, and GPU‑enabled LoRA pipeline on a shared Docker network.

## Prerequisites

- **Python** – 3.12+ (for the backend).
- **Node.js** – 18+ (for the Vite frontend).
- **Docker & Compose** – Recommended for running the full stack and GPU‑accelerated LoRA.
- **GPU (optional but ideal)** – Nvidia GPU if you want to run the FIBO LoRA pipeline locally.
- **API keys / secrets** (configured via `.env`):
  - `OPENAI_API_KEY`, `OPENAI_MODEL_NAME`, `OPENAI_MODEL_INSTRUCTION` – For the agent and palette generator.
  - `IMAGE_STUDIO_KEY`, `IMAGE_STUDIO_ENDPOINT` – For BRIA Vision Studio / FIBO.
  - `GOOGLE_API_KEY` – For Gemini JSON prompt generation.
  - `HUGGINGFACE_HUB_TOKEN` – For model downloads used by segmentation / LoRA.
  - `GAZE_TRACKER_MCP_PORT` – Port for the gaze MCP server.
  - `LORA_PATH` – Path (or mounted volume) containing the trained LoRA weights.

## Setup

1. **Clone the repo**

   ```bash
   git clone <repo-url> fibo-brand-studio
   cd fibo-brand-studio
   ```

2. **Create an `.env`**

   Start from the backend template and promote it to a shared `.env` at the repo root so both Docker Compose and the backend see the same values:

   ```bash
   cp backend/.env.dist .env
   ```

   Then open `.env` and fill in:

   - Backend keys like `OPENAI_API_KEY`, `IMAGE_STUDIO_KEY`, `IMAGE_STUDIO_ENDPOINT`, `GOOGLE_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, `GAZE_TRACKER_MCP_PORT`, `LORA_PATH`, `TESTING_ENVIRONMENT`.
   - Compose / port settings, for example:
     ```bash
     FRONTEND_CONTAINER=lovable-frontend
     FRONTEND_PORT=5173
     FRONTEND_DEBUGGER_PORT=9320
     BACKEND_CONTAINER=fibo-backend
     VITE_BACKEND_HOST=localhost
     VITE_BACKEND_PORT=8000
     BACKEND_DEBUGGER_PORT=5678
     CHECKPOINTS_URL=sqlite:///checkpoints.db
     OPENAI_MODEL_NAME=gpt-4.1-mini
     OPENAI_MODEL_INSTRUCTION="You are a helpful brand-aware image agent."
     ```

3. **Backend (Python, local dev)**

   ```bash
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

   # Run the FastAPI app (without Docker)
   uvicorn main_fastapi:app --host 0.0.0.0 --port "${VITE_BACKEND_PORT:-8000}"
   ```

   This starts:
   - The `/ask` endpoint for agentic chats.
   - The `/segment` endpoint for segmentation + palette extraction.
   - The `/download_temp_img/{img_path}` endpoint used by tools to expose temp images.

4. **Frontend (Vite + React)**

   ```bash
   cd lovable-frontend
   npm install          # or pnpm install / bun install
   npm run dev          # serves on FRONTEND_PORT (default 5173)
   ```

   Point the frontend’s API base URL at the backend, e.g. `VITE_API_BASE_URL=http://localhost:8000` in the frontend’s `.env` if needed.

5. **Full stack via Docker Compose**

   Once your `.env` is configured and a Docker network exists:

   ```bash
   docker network create net || true
   docker compose up --build
   ```

   - The backend container runs on `VITE_BACKEND_PORT` and mounts the LoRA weights from `LORA_PATH`.
   - The frontend container runs on `FRONTEND_PORT` and hot‑reloads when you edit files thanks to bind mounts.

## Example: call the agent from the CLI

With the backend running (locally or in Docker), you can trigger the workflow directly:

```bash
curl -X POST "http://localhost:${VITE_BACKEND_PORT:-8000}/ask" \
  -H "Content-Type: application/json" \
  -d '{
        "chat_id": "demo-thread-1",
        "query": "Design a hero shot for the FIBO launch page that uses our brand pink #EA1E63 and yellow #FFCA3A, with the mascot looking toward the camera.",
        "seed": 1234
      }'
```

The response includes:

- Natural‑language agent output under `response.content`.
- An `image_url` field pointing to the generated image or overlay.
- A `structured_prompt` JSON payload (when LoRA/JSON mode is used).
- The `seed` used so you can reproduce or refine the result.

This is the same flow the frontend uses, and it showcases the JSON‑native, agentic loop that makes this a strong fit for **Best JSON‑Native or Agentic Workflow**.

