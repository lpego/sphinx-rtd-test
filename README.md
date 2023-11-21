![deploy_docs](https://github.com/lpego/sphinx-rtd-test.yaml/actions/workflows/build_sphinx_docs/badge.svg)

# sphinx-rtd-test
Testing deployment of Sphinx-generated documentation to ReadTheDocs (perhaps GitHub Pages too)

Both `sphinx_req.txt` and `sphinx_env.yaml` install a base environment without any mzbsuite dependencies (only Sphinx). 

`mzbfull_req.txt` and `mzbfull_env.yaml` install full environments with all dependencies (necessary in order for sphinx_automodule to compile without errors). 

`mzbsuite_env.yaml` is a manually curated env for installing all dependencies (cna probably remove some libraries still). 

## GitHub Actions / Pages

In the directory `.github/workflows` there's a file `deploy_docs.yaml` that contains the job instructions to install and run sphinx to build the docs. 

The GitHub Page where the Sphinx docs are served should be located at: https://lpego.github.io/sphinx-rtd-test/

Check GitHub Action status here: https://github.com/lpego/sphinx-rtd-test/actions

## ReadTheDocs

The page is located at: https://lpego-sphinx-rtd-test.readthedocs.io/en/latest/

# Where we got so far

- [x] Built minimal environment for Sphinx
- [x] Built full environment with all mzbsuite dependencies
- [x] Built Sphinx documentation
    - [x] with no errors
    - [x] with no warnings
    - [x] notebooks render
    - [ ] notebooks compile
- [x] GitHub Pages deployment
    - [x] found GitHub action to make Pandoc available in `$PATH`
    - [x] implemented GitHub action to setup micromamba
    - [x] successfully created conda env from YAML
    - [x] figured out how to activate newly created env to run `sphinx-build` in it
    - [x] Sphinx build succeeds (with the same warning as local runs)
    - [x] figured out why no files in branch `gh-pages`
    - [x] Deployment to gh-pages works
    - [x] GitHub Pages website serving successfull: 
- [x] ReadTheDocs deployment 
    - [x] made custom `mzbsuite_env.txt` requirements file for RtD build
    - [x] env successfully built on RtD
    - [x] build completes
    - [x] docs are served on webpage

# ToDo 

- [ ] figure out notebook compilation... (currently disabled)