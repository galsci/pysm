PYTHON ?= python
TEST_FILE := test_high_flux_catalog_script.py

# Run the local test bypassing project-wide nbval addopts/plugins
.PHONY: test
test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PYTHON) -m pytest -q -c /dev/null $(TEST_FILE)

# Clean temporary pytest artifacts (none normally created due to -c /dev/null)
.PHONY: clean
clean:
	@echo "Nothing to clean."
