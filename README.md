# sphinx-rtd-test
Testing deployment of Sphinx-generated documentation to ReadTheDocs (perhaps GitHub Pages too)

Both `sphinx_req.txt` and `sphinx_env.yaml` install a base environment without any mzbsuite dependencies (only Sphinx). 

`mzbfull_req.txt` and `mzbfull_env.yaml` install full environments with all dependencies (necessary in order for sphinx_automodule to compile without errors). 

## GitHub Actions / Pages

In the directory `.github/workflows` there's a file `deploy_docs.yaml` that contains the job instructions to install and run sphinx to build the docs. 

The GitHub Page where the Sphinx docs are served should be located at: https://lpego.github.io/sphinx-rtd-test/

Check GitHub Action status here: https://github.com/lpego/sphinx-rtd-test/actions

# Where we got so far

- [x] Built minimal environment for Sphinx
- [x] Built full environment with all mzbsuite dependencies
- [x] Built Sphinx documentation
    - [x] with no errors
    - [x] with no warnings
    - [x] notebooks render
    - [ ] notebooks compile
- [ ] GitHub Pages deployment
    - [x] found GitHub action to make Pandoc available in `$PATH`
    - [x] implemented GitHub action to setup micromamba
    - [x] successfully created conda env from YAML
    - [] trying to activate newly created env to run `sphinx-build` in it
- [ ] ReadTheDocs deployment 
    - cannot use conda to install packages... 

# ToDo 

[ ] figure out notebook compilation... (currently disabled)