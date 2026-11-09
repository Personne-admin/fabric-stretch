# Tracker tooling

See the [tracker guide](../docs/self_hosting_tracker.md) for setup, commands and troubleshooting.
Defaults and version pins are in [`tracker.env.example`](tracker.env.example).

`fabsim-track` is the frontend, `bootstrap.sh` installs the environment, `lib/` holds the helpers,
and `tests/` uses a fake tracker. `Dockerfile` builds the container.
