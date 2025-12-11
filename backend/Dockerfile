#####################
# First stage: build
FROM python:3.12 AS build

RUN python3.12 -m venv .venv
ENV PATH="/.venv/bin:$PATH"

# Dependencies
RUN pip install uv

# SAM3
RUN uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN git clone https://github.com/facebookresearch/sam3.git sam3
WORKDIR /sam3
RUN uv pip install -e ".[train,dev,notebooks]"

WORKDIR /
COPY pyproject.toml .
RUN uv pip install -r pyproject.toml --all-extras --prerelease=allow 

#####################
# Second stage: runtime
FROM python:3.12-slim AS runtime

COPY --from=build .venv .venv
COPY --from=build sam3 sam3
COPY . . 

ENV PATH=".venv/bin:$PATH"
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]