# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-11-23

### Fixed
- ✅ Recreated virtual environment with proper Python 3.13 support
- ✅ Fixed all dependency installations
- ✅ Verified project can run successfully
- ✅ Fixed import paths and module dependencies

### Added
- ✅ Comprehensive test suite with 15 passing tests
  - Basic functionality tests
  - Propagation engine tests
  - AP placement algorithm tests
  - Material system tests
- ✅ Test coverage infrastructure (pytest, pytest-cov)
- ✅ Quick setup guide (README_SETUP.md)
- ✅ Testing documentation (TESTING.md)
- ✅ Architecture documentation (ARCHITECTURE.md)
- ✅ Changelog (this file)
- ✅ Updated .gitignore with comprehensive exclusions

### Removed
- ❌ Unused Node.js dependencies (package.json, package-lock.json)
- ❌ No React frontend exists, so removed misleading dependencies

### Changed
- 📝 Updated documentation structure
- 📝 Improved project organization
- 📝 Enhanced error handling in tests

### Testing
- 15/15 tests passing (100% pass rate)
- Core functionality validated
- Propagation engines verified
- AP placement algorithms tested

### Known Issues
- COST-231 propagation model has non-monotonic behavior at very short distances (< 5m)
- This is a known characteristic of the model, not a bug

## [1.0.0] - 2024

### Initial Release
- WiFi signal strength prediction
- AP placement optimization
- Multiple propagation models
- Material-aware signal modeling
- 3D visualization
- GUI applications
- Machine learning integration
