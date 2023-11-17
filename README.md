# sphinx-rtd-test
Testing deployment of Sphinx-generated documentation to ReadTheDocs (perhaps GitHub Pages too)

Both `sphinx_req.txt` and `sphinx_env.yaml` install a base environment without any mzbsuite dependencies (only Sphinx). 

`mzbfull_req.txt` and `mzbfull_env.yaml` install full environments with all dependencies (necessary in order for sphinx_automodule to compile without errors). 