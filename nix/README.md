# Building with NiX

## What is Nix?

Nix is a powerful package manager and build system that provides reproducible
builds, declarative configurations, and isolated development environments. It
solves common problems in software development, such as dependency conflicts,
non-reproducible builds, and difficulty in managing multiple versions of
dependencies.

### Key Features of Nix:

- **Reproducibility**: Ensures that builds are consistent across different
  systems.
- **Isolation**: Dependencies are isolated, preventing conflicts between
  projects.
- **Versioning**: Allows managing multiple versions of dependencies and tools.
- **Declarative Configuration**: Environments and builds are defined in a single
  configuration file.

## How We Use Nix in This Project

In this project, Nix is used to manage dependencies and build environments for
both development and specific versions of the project. This ensures that all
contributors and users can work with the same setup, regardless of their
operating system or local configurations.

### Developing OpenPFC

To set up the development environment, where `HeFFTe` is built from its
respective `master` branch and OpenPFC is built from the local source, run:

```bash
nix develop
```

This will create a shell environment with the latest source code for both
projects. Inside the shell, you can build the project using the following
commands:

```bash
cmake -S . -B build
cmake --build build
```

### Building OpenPFC

The build targets follow the pattern:

```
#openpfc-<what>-<version>
```

- `<what>` can be one of the following:
  - Empty: Builds the full release version.
  - `apps`: Builds only the apps.
  - `docs`: Builds only the documentation.
  - `tests`: Builds only the tests.
  - `examples`: Builds only the examples.
- `<version>` specifies the version number without the `v` prefix. Special values:
  - `dev`: Builds from the local source directory (defaults to `Debug` build type).
  - `master`: Builds the bleeding-edge version from the GitHub repository (defaults to `Release` build type).

### Overriding the Build Type

By default:
- `dev` builds use the `Debug` build type.
- All other builds use the `Release` build type.

You can override this behavior by passing the `--arg buildType <type>` argument, where `<type>` can be `Debug` or `Release`.

For example:
```bash
nix build .#openpfc --arg buildType '"Debug"'
```

### Examples

Here are some examples of how to use the build targets:

- **Build the default release version**:
  ```bash
  nix build .#openpfc
  ```

- **Build the development version from the source directory**:
  ```bash
  nix build .#openpfc-dev
  ```

- **Build a specific version (e.g., 0.1.1)**:
  ```bash
  nix build .#openpfc-0.1.1
  ```

- **Build tests for a specific version (e.g., 0.1.0)**:
  ```bash
  nix build .#openpfc-tests-0.1.0
  ```

- **Build documentation for a specific version (e.g., 0.1.1)**:
  ```bash
  nix build .#openpfc-docs-0.1.1
  ```

- **Build documentation for the bleeding-edge version**:
  ```bash
  nix build .#openpfc-docs-master
  ```

- **Build tests for the development version**:
  ```bash
  nix build .#openpfc-tests-dev
  ```

- **Override the build type to Debug for a release version**:
  ```bash
  nix build .#openpfc-0.1.1 --arg buildType '"Debug"'
  ```

These are all the build targets available in this project so far:

```
nix build .#openpfc
nix build .#openpfc-0.1.0
nix build .#openpfc-0.1.1
nix build .#openpfc-master
nix build .#openpfc-dev

nix build .#openpfc-tests
nix build .#openpfc-tests-0.1.0
nix build .#openpfc-tests-0.1.1
nix build .#openpfc-tests-master
nix build .#openpfc-tests-dev

nix build .#openpfc-docs
nix build .#openpfc-docs-0.1.0
nix build .#openpfc-docs-0.1.1
nix build .#openpfc-docs-master
nix build .#openpfc-docs-dev

nix build .#openpfc-apps
nix build .#openpfc-apps-0.1.0
nix build .#openpfc-apps-0.1.1
nix build .#openpfc-apps-master
nix build .#openpfc-apps-dev

nix build .#openpfc-examples
nix build .#openpfc-examples-0.1.0
nix build .#openpfc-examples-0.1.1
nix build .#openpfc-examples-master
nix build .#openpfc-examples-dev
```

### Why Use Nix?

By using Nix, we ensure that:

- All contributors have a consistent development environment.
- Builds are reproducible and reliable.
- Managing dependencies and versions is straightforward and conflict-free.

For more information about Nix, visit the [official
documentation](https://nixos.org/manual/nix/stable/).
