# Debian Packaging Guide for python_magnetcooling

This directory contains the Debian packaging files for `python_magnetcooling`.

## Package Information

- **Source Package**: python-magnetcooling
- **Binary Package**: python3-magnetcooling
- **Version**: 0.1.0-1
- **Architecture**: all (pure Python)
- **Section**: python
- **Priority**: optional

## Files Description

### Required Files

- **control**: Package metadata, dependencies, and description
- **rules**: Build instructions for the package
- **changelog**: Version history and changes
- **copyright**: Licensing information (MIT License)
- **compat**: Debhelper compatibility level (13)
- **source/format**: Source package format (3.0 native)

### Optional Files

- **python3-magnetcooling.install**: Specifies which files to install and where
- **README.Debian**: Debian-specific documentation for users
- **watch**: Monitors upstream releases for new versions
- **source/options**: Build options

## Building the Package

### Prerequisites

Install the required build tools:

```bash
sudo apt-get install debhelper dh-python python3-all python3-setuptools \
                     devscripts build-essential
```

Install build dependencies:

```bash
sudo apt-get install python3-numpy python3-scipy python3-pandas \
                     python3-iapws python3-pint python3-ht \
                     python3-pytest python3-pytest-cov
```

Note: Some dependencies might not be available in standard repositories. You may need to:
- Build them from source
- Use pip to install them (not recommended for production packages)
- Add additional repositories

### Build Methods

#### Method 1: Using the build script (Recommended)

```bash
./build-deb.sh
```

#### Method 2: Manual build

```bash
# Build binary package only
dpkg-buildpackage -us -uc -b

# Build source and binary packages
dpkg-buildpackage -us -uc

# Build with signature (requires GPG key)
dpkg-buildpackage
```

#### Method 3: Using pbuilder (clean environment)

```bash
# Create pbuilder environment (first time only)
sudo pbuilder create

# Build package
sudo pbuilder build ../python-magnetcooling_*.dsc
```

### Build Output

After a successful build, the following files will be created in the parent directory:

- `python3-magnetcooling_0.1.0-1_all.deb` - The binary package
- `python-magnetcooling_0.1.0-1.buildinfo` - Build information
- `python-magnetcooling_0.1.0-1_amd64.changes` - Changes file
- `python-magnetcooling_0.1.0-1.dsc` - Source package description (if built)
- `python-magnetcooling_0.1.0-1.tar.xz` - Source tarball (if built)

## Installing the Package

### From the built .deb file

```bash
sudo dpkg -i ../python3-magnetcooling_0.1.0-1_all.deb
sudo apt-get install -f  # Fix any dependency issues
```

### Removing the package

```bash
sudo apt-get remove python3-magnetcooling
# or to remove including configuration files
sudo apt-get purge python3-magnetcooling
```

## Package Contents

After installation, the package will install:

- Python module: `/usr/lib/python3/dist-packages/python_magnetcooling/`
- Examples: `/usr/share/doc/python3-magnetcooling/examples/`
- Documentation: `/usr/share/doc/python3-magnetcooling/`

## Lintian Checks

To check the package for common issues:

```bash
lintian ../python3-magnetcooling_0.1.0-1_all.deb
```

## Updating the Package

### For a new upstream version:

1. Update `debian/changelog`:
   ```bash
   dch -v 0.2.0-1 "New upstream release"
   ```

2. Update version in `pyproject.toml` if needed

3. Rebuild the package

### For packaging fixes (same upstream version):

```bash
dch -i "Fixed packaging issue"
```

This will increment the Debian revision (e.g., 0.1.0-1 → 0.1.0-2)

## Uploading to a Repository

### To a personal repository:

```bash
# Sign the changes file
debsign ../python-magnetcooling_0.1.0-1_amd64.changes

# Upload using dput (configure your repository first)
dput my-repo ../python-magnetcooling_0.1.0-1_amd64.changes
```

### Creating a local repository:

```bash
# Install reprepro
sudo apt-get install reprepro

# Set up a repository (in a separate directory)
mkdir -p ~/debian-repo
cd ~/debian-repo
# Create conf/distributions file
# Add the package
reprepro includedeb unstable /path/to/python3-magnetcooling_0.1.0-1_all.deb
```

## Common Issues

### Missing Dependencies

Some Python dependencies might not be packaged for Debian. Options:

1. Package them yourself (recommended for production)
2. Use pip to install them system-wide (not recommended)
3. Relax version requirements in `debian/control`

### Build Failures

- Check `debian/rules` for correct build system
- Ensure all build dependencies are installed
- Review build logs for specific errors

### Test Failures

Tests can be skipped during build by modifying `debian/rules`:

```makefile
override_dh_auto_test:
	# Skip tests
```

## Standards Compliance

This package follows:
- Debian Policy Manual version 4.6.2
- Python Debian packaging guidelines
- debhelper compatibility level 13

## Maintainer Information

**Maintainer**: Christophe Trophime <christophe.trophime@lncmi.cnrs.fr>

For packaging issues, please check:
- [Debian Python Policy](https://www.debian.org/doc/packaging-manuals/python-policy/)
- [Debian New Maintainers' Guide](https://www.debian.org/doc/manuals/maint-guide/)
- [debhelper documentation](https://manpages.debian.org/testing/debhelper/debhelper.7.en.html)
