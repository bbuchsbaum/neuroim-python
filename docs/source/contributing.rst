Contributing to PyNeuroim
=========================

Thank you for your interest in contributing to PyNeuroim! This guide will help you get started.

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally::

    git clone https://github.com/YOUR_USERNAME/neuroimpy.git
    cd neuroimpy

3. Create a virtual environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install in development mode::

    pip install -e ".[dev]"

Development Workflow
--------------------

1. Create a new branch for your feature or bugfix::

    git checkout -b feature-name

2. Make your changes and write tests
3. Run the test suite::

    pytest

4. Check code style::

    black src/ tests/
    flake8 src/ tests/
    mypy src/

5. Commit your changes::

    git add .
    git commit -m "Add feature: brief description"

6. Push to your fork and create a pull request

Code Style Guide
----------------

We use the following tools to maintain code quality:

- **Black** for code formatting (line length: 88)
- **Flake8** for linting
- **MyPy** for type checking

All code should:

- Follow PEP 8 style guidelines
- Include type hints where appropriate
- Have docstrings for all public functions and classes
- Include unit tests for new functionality

Docstring Format
~~~~~~~~~~~~~~~~

We use NumPy-style docstrings::

    def function_name(param1: type1, param2: type2) -> return_type:
        """Brief description of function.

        Longer description if needed.

        Parameters
        ----------
        param1 : type1
            Description of param1
        param2 : type2
            Description of param2

        Returns
        -------
        return_type
            Description of return value

        Examples
        --------
        >>> function_name(arg1, arg2)
        expected_output
        """

Testing Guidelines
------------------

- Write tests for all new functionality
- Place tests in the ``tests/`` directory
- Use pytest fixtures for common test data
- Aim for at least 80% code coverage
- Test edge cases and error conditions

Example test::

    def test_neurovol_arithmetic():
        """Test arithmetic operations on NeuroVol."""
        space = pn.NeuroSpace(dim=(10, 10, 10))
        data = np.ones((10, 10, 10))
        vol = pn.NeuroVol(data, space)

        # Test addition
        result = vol + 5
        assert np.all(result.data == 6)

Performance Considerations
--------------------------

When working with neuroimaging data:

- Use NumPy operations instead of Python loops
- Consider memory usage for large datasets
- Provide sparse/lazy options where appropriate
- Profile performance-critical code

Documentation
-------------

- Update docstrings for any modified functions
- Add examples to docstrings
- Update relevant .rst files in ``docs/source/``
- Build docs locally to check formatting::

    cd docs
    make html

Pull Request Process
--------------------

1. Ensure all tests pass
2. Update documentation as needed
3. Add entry to CHANGELOG.rst
4. Reference any related issues
5. Request review from maintainers

Your PR should:

- Have a clear, descriptive title
- Include a summary of changes
- Reference related issues with "Fixes #123"
- Pass all CI checks

Reporting Issues
----------------

When reporting issues, please include:

- Python version and OS
- PyNeuroim version
- Minimal code to reproduce the issue
- Full error traceback
- Expected vs actual behavior

Feature Requests
----------------

We welcome feature requests! Please:

- Check existing issues first
- Describe the use case
- Provide examples if possible
- Consider implementing it yourself!

Community Guidelines
--------------------

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Follow the `Code of Conduct <https://github.com/bbuchsbaum/neuroimpy/blob/main/CODE_OF_CONDUCT.md>`_

Questions?
----------

- Open a discussion on GitHub
- Check the documentation
- Look at existing issues

Thank you for contributing to PyNeuroim!
