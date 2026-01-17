# Planning

## Purpose
This repository aims to provide a reproducible, scriptable simulation tool for railway to structure impact scenarios, with a focus on a stable CLI, auditable configuration, and machine readable outputs.

## Working agreement
1. Work in small increments.
2. Pick exactly one next task from docs/TASKS.md.
3. Make minimal code changes that are consistent with the current architecture.
4. Verify with an explicit command or test.
5. Update docs/TASKS.md and docs/WORKLOG.md every time.

## Quality gates
1. A minimal CLI run produces results.csv and a log in the output directory.
2. pytest passes, or a clear explanation exists in TASKS for any skipped or xfailed tests.
3. Documentation examples reference files that exist in the repository.

## Repo orientation
1. src/railway_simulator contains the core library.
2. configs contains example scenario configurations.
3. tests contains regression and unit tests.
4. docs contains project process docs and technical notes.

## Definition of done
A task is done when:
1. The expected user facing behavior is achieved.
2. A verification command is recorded.
3. TASKS and WORKLOG are updated.
