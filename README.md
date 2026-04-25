# hlh-server

Standalone private server repo for Humanity's Last Hackathon.

This repo exposes:

- `GET /health`
- `POST /v1/hlh/profile`
- `POST /v1/hlh/submit`
- `POST /v1/hlh/submit/bundle`

## Requirements

- Python 3.12+
- `popcorn` CLI installed and authenticated on the server host with `popcorn auth`
- Hugging Face credentials configured on the server host when bundle upload is enabled
- Apple Silicon with MPS available for `profile`

## Install

```bash
pip install -e .
```

## Environment

```bash
export HLH_SERVICE_HOST=127.0.0.1
export HLH_SERVICE_PORT=8788
```

Configure server-side credentials before running the service:

```bash
popcorn auth
export HF_TOKEN=hf_...
```

`popcorn auth` is required because the service validates incoming `X-Popcorn-Cli-Id` values against Popcorn and runs `popcorn submit` server-side. `HF_TOKEN` is required when `HLH_SUBMIT_BUNDLE_ENABLED=1`.

Optional bundle upload configuration:

```bash
export HLH_UPLOAD_TOP_SUBMISSION_BUNDLE=0
export HLH_SUBMIT_BUNDLE_ENABLED=1
export HLH_SUBMIT_BUNDLE_BUCKET_TEMPLATE="{hf_user}/humanitys-last-hackathon"
export HLH_SUBMIT_BUNDLE_HF_TOKEN_ENV=HF_TOKEN
```

Optional GitHub-backed profile runner:

```bash
export HLH_PROFILE_RUNNER=github
export HLH_PROFILE_WORKFLOW_FILE=main.yml
export HLH_PROFILE_WORKFLOW_REF=main
export HLH_PROFILE_POLL_SECONDS=5
export HLH_PROFILE_TIMEOUT_SECONDS=600
```

`HLH_PROFILE_RUNNER=service` is the default and profiles on the server host directly.
`HLH_PROFILE_RUNNER=github` dispatches the bundled GitHub Actions workflow through `gh`.

## Run

```bash
hlh-server
```

## Notes

- `profile` can run either on the server host or through GitHub Actions, depending on `HLH_PROFILE_RUNNER`.
- `submit` runs `popcorn submit` server-side and can optionally request a follow-up bundle upload.
- The GitHub-backed profile path requires authenticated `gh` CLI access with permission to dispatch and inspect workflow runs for this repo.
