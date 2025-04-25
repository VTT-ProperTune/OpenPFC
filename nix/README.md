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

### Building Specific Releases

To build a specific release of `OpenPFC` with a specific version of `HeFFTe`,
use the `nix build` command. There are two main build targets:

- `#openpfc-dev`: The default target, which can also be invoked with `nix
  build`. This builds the development version of `OpenPFC`.
- `#openpfc`: Builds the release version of `OpenPFC`.

For example:

```bash
nix build #openpfc
```

or equivalently for the development version:

```bash
nix build #openpfc-dev
```

When building, tagged versions are used, which are defined in the following files:

- `nix/openpfc/versions`
- `nix/heffte/versions`

This approach allows constructing immutable builds for all versions simply by
changing the version numbers in these files.

### Why Use Nix?

By using Nix, we ensure that:

- All contributors have a consistent development environment.
- Builds are reproducible and reliable.
- Managing dependencies and versions is straightforward and conflict-free.

For more information about Nix, visit the [official
documentation](https://nixos.org/manual/nix/stable/).
