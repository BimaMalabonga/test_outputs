# [Project Name]

## Setup

1. In the terminal, run the project init script:

    ```cmd
    python init PROJECT_NAME

    # eg: python init Nomination-Package
    ```
  
2. In the terminal, update the Poetry .lock file:

    ```cmd
    poetry update
    ```
  
3. Update description and authors in `pyproject.toml`. For authors, provide both the name and the
DET email address. Example: `authors = ["John Doe <j.doe@det.nl>"]`.
4. Update the `README.md` file:
   - Replace the top header ("[Project name]") with the model name
   - Provide the high-level purpose of the model
   - Delete the "Setup" section (the "Development settings" section can be left unchanged)

Before any coding, merge all changes from steps 1-4 in a single, dedicated pull request.

## Configuration

### Project Structure

The project mainly follows the standard Poetry structure:

```
project
├── pyproject.toml
├── README.md
├── run.py
├── tasks.py
├── src
│   ├── __init__.py
│   └── main.py
└── tests
    └── __init__.py
```

The entry point of the script is the run.py module. Its content should not be changed.

The main function of the script is the `main()` function of the ./src/main.py module.

The entire codebase should be placed in the src folder.

### Environment Variables

Environment variables can be specified in a .env file located in the project's root directory.

This file can be used e.g. to store credentials and sensitive information that are needed to
run the script.

The content of the file is loaded in the run.py module.

### Dependencies

#### Dependency Manager

Project dependencies are managed by [Poetry](https://python-poetry.org/).

#### Dependabot

Automated dependency updates are executed with
[Dependabot](https://docs.github.com/en/code-security/dependabot).

### Invoke Tasks

The project uses the [Invoke](https://www.pyinvoke.org/) package to simplify certain tasks such
as testing and code formatting.

All tasks are defined in a tasks.py file.

Invoke tasks can be executed directly from the terminal, using the `inv` (or `invoke`) command
line tool.

For guidance on the available Invoke tasks, execute the following command in the terminal:

```cmd
inv --list
```

Use the `-h` (or `--help`) argument for help about a particular Invoke task. For example:

```cmd
inv lint -h
```

### Testing

The project contains the infrastructure for unit testing and snapshot testing.

#### General Procedure

- All code changes are tested with the [Pytest](https://github.com/pytest-dev/pytest) package.
- Pytest requires test modules to follow the naming convention 'test_*.py'.
- Pytest requires test functions to follow the naming convention 'test_*()'.

#### Unit Testing

Unit tests can be implemented directly in the 'test_unit.py' module.

Unit tests can be run using the following command in the terminal:

```cmd
inv run-unit-test
```

#### Snapshot Testing

Snapshot tests verify that model outputs do not change when model inputs do not change.

Snapshot testing is typically done with multiple sets of inputs, covering different possible
(combinations of) model settings. One set of inputs and its corresponding expected model outputs
constitute a _test case_. Each test case needs to be stored in a dedicated folder named
'Case_[XX]' according to the following logic:

```
tests/
└── snapshots/
    ├── Case01
    │  ├── Inputs
    │  └── ExpectedOutputs
    ├── Case02
    │  ├── Inputs
    │  └── ExpectedOutputs
    └──  ...
```

**Creating new test cases:**

New test cases (including inputs and expected outputs) can be added manually. Alternatively, they
can easily be created based on the current content of the project's 'Inputs' folder, using the
following command in the terminal:

```cmd
inv run-snapshot-test -t create
```

**Updating the expected outputs of existing test cases:**

To update the expected outputs of all test cases, execute the following command in the terminal:

```cmd
inv run-snapshot-test -t update
```

**Running snapshot tests:**

Snapshot tests can be run using the following command in the terminal:

```cmd
inv run-snapshot-test
```

### GitHub Actions

The project's CI/CD pipeline is enforced with [GitHub actions](https://docs.github.com/en/actions)
workflows.

#### Workflow: Continuous Integration (CI)

The continuous integration (CI) workflow runs tests to check the integrity of the codebase's
content, and linters to check the consistency of its format.

The workflow was inspired by the following preconfigured templates:

- [Python package](https://github.com/actions/starter-workflows/blob/main/ci/python-package.yml):
  A general workflow template for Python packages.
- [Poetry action](https://github.com/marketplace/actions/install-poetry-action): A GitHub action
  for installing and configuring Poetry.

##### CI Check: Testing

The CI check is executed with the following Invoke task:

```cmd
inv test -c
```

If the CI check fails because of differences in the outputs of snapshot tests, the expected
outputs can be updated using the following command in the terminal:

```cmd
inv run-snapshot-test -t update
```

##### CI Check: Code Formatting

Linters are used to check that the code is properly formatted:

- [Isort](https://github.com/timothycrosley/isort) for the imports section
- [Darglint](https://github.com/terrencepreilly/darglint) for the docstrings description
- [Black](https://github.com/psf/black) for the main code
- [Pymarkdown](https://github.com/jackdewinter/pymarkdown) for the markdown file README.md

The CI check is executed with the following Invoke task:

```cmd
inv lint -c
```

If the CI check fails, execute the following command in the terminal:

```cmd
inv lint
```

This command fixes the parts of the code that should be reformatted. Adding the `-c` (or
`--check`) optional argument instructs the command to only _check_ if parts of the code should be
reformatted, without applying any actual changes.
