# AGENTS.md

## Project Purpose

This repository is a minimal decision-making prototype for robotics-style planning.
The codebase should stay intentionally small, modular, and easy to reason about.
We are building the system in phases, with each phase delivering a concrete, testable step forward.

## Core Development Rules

- Use Python only.
- Always run Python commands with `.venv/bin/python`.
- Prefer simple, explicit implementations over abstractions.
- Split code across small modules with clear responsibilities.
- Complete and test each phase before moving on to the next one.
- Print logs that make reasoning and debugging easy to follow.
- Do not add unnecessary dependencies.
- Preserve readability, minimalism, and directness in every change.

## Repo Layout

The repository is intentionally lightweight and will grow in small steps.

- `AGENTS.md`: repository-level development guide and working agreement.
- `requirements.txt`: pinned Python dependencies for the project.
- `test_env.py`: environment verification script and simple execution smoke test.
- `.venv/`: local virtual environment for development only; never rely on system Python for project commands.

As the project grows, prefer layouts like:

- `src/` for runtime modules
- `tests/` for automated tests
- `scripts/` for explicit developer utilities

Keep modules small. If a file starts doing more than one clear job, split it.

## Running The Project

Create or refresh the environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the current project script:

```bash
.venv/bin/python test_env.py
```

If a main entry point is added later, document and run it the same way:

```bash
.venv/bin/python path/to_entrypoint.py
```

## Running Tests

If tests are added, run them with the virtual environment Python only.

Examples:

```bash
.venv/bin/python -m pytest
```

or, for standard library tests:

```bash
.venv/bin/python -m unittest discover
```

Prefer simple test commands with explicit output. Keep tests close to the behavior they verify.

## Phase-Based Delivery

Work in phases.
Each phase should:

- implement one bounded capability,
- include enough logging to inspect reasoning and behavior,
- be verified with a smoke test or automated test,
- leave the repository in a readable and stable state.

Do not begin the next phase until the current phase is working and tested.

## Logging Expectations

Reasoning and debugging information should be printed to the console.
Use logs to show:

- what the system is deciding,
- what inputs it received,
- what outputs or actions it produced,
- where a failure or unexpected branch occurred.

Logs should help a human understand behavior quickly without adding heavy logging frameworks unless clearly necessary.

## Dependency Policy

Every dependency must have a clear purpose.
Before adding one, prefer:

- the Python standard library,
- a small explicit helper module in this repo,
- a simpler implementation that avoids the dependency entirely.

If a dependency is added, keep it minimal, justify it in the change, and update `requirements.txt`.

## Definition Of Done

A task in this repository is done when:

- the requested behavior is implemented,
- the implementation stays small, explicit, and readable,
- code is split sensibly across small modules,
- reasoning/debug logs are present where they help inspection,
- the phase is verified by running the relevant script or tests with `.venv/bin/python`,
- no unnecessary dependencies or abstractions were introduced,
- any new entry points, modules, or tests are easy for the next contributor to find and run.
