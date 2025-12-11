source .venv/bin/activate


if [[ "$DEBUG_BACKEND" == "1" ]]; then
  echo "Debug mode activated with debugpy on port ${BACKEND_DEBUGGER_PORT}"
  echo "Start the debugger to run the docker image on port ${BACKEND_PORT}."

  python -m debugpy --listen 0.0.0.0:$BACKEND_DEBUGGER_PORT --wait-for-client \
    -m uvicorn main_fastapi:app --host 0.0.0.0 --port "${BACKEND_PORT}" --reload --loop asyncio

else
  uvicorn main_fastapi:app --host 0.0.0.0 --port "${BACKEND_PORT}"

fi