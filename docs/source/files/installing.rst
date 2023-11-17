Installation
============

The Project contains two main parts: 
    1. The ``mzbsuite`` package; 
    2. The ``scripts`` and ``workflow`` files making use of its functions. 

The ``mzbsuite`` folder (in Python this is called a package) contains higher-level functions that can be imported from other scripts and used to create complex processing pipelines, while the ``scripts`` and ``workflow`` folders contain files that make use of functions implemented in ``mzbsuite`` to run single modules or the processing pipeline as a whole. 

The file ``environment.yml`` contains all the minimal dependencies for the project, and should install the functions in ``mzbsuite`` as well. However, if this does not work, the ``mzbsuite`` package can be installed separately using the ``setup.py`` file in the ``mzbsuite`` folder, via ``pip`` (see :ref:`here <pip_install_mzbsuite>`)

Docker container
-------------------
For less experienced users, we recommend installing via Docker. You can find instructions on how to install the Docker Engine here: `<https://docs.docker.com/get-docker/>`_

Once the Docker Engine has been correctly installed, you can download the project's Docker image from: 

    .. class:: 

    `*link coming soon* <LINK_TO_DOCKER_IMAGE>`_ 

and launch it from the Graphical User Interface (GUI) of Docker. 

This should open a terminal within the Docker environment where you can start using the project's modules immediately! 

Manual install
--------------

If you prefer to install the project directly in your local environment or just want to use the functions in your own scripts, you can download the project's  repository. 
The project is currently hosted on the `Swiss Data Science Center <https://datascience.ch>`_ GitLab server, you can find the repository here: 

    .. class:: 

    `<https://gitlab.renkulab.io/biodetect/mzb-workflow>`_. 

To download the project, you simply need to clone it into a location of your choice: 

.. code-block:: bash

    git clone git@renkulab.io:biodetect/mzb-workflow.git

.. admonition:: \ \ 
   
   If you don't have Git installed, you can follow instructions `here <https://git-scm.com/downloads>`_. We recommend using Git because it allows to easily update the package and tracking any changes you make. 

This will create a folder called ``mzb-workflow`` in the current working directory. 

.. hint:: \ \ 
   If you don't want to use Git, you can directly download an archive of the `repository <https://renkulab.io/gitlab/biodetect/mzb-workflow>`_ from GitLab and extract it manually. 

You can then install the necessary packages using the conda package manager and the ``environment.yml`` file: 

.. code-block:: bash

    cd mzb-workflow    # chdir mzb-workflow in Windows
    conda env create -f environment.yml

.. hint:: \ \ 
   If you don't have Anaconda installed, you can can get the installer from `<https://www.anaconda.com/download>`_.

This should install the ``mzbsuite`` package as well, but if this does not work, you can simply install it via pip as: 

.. _pip_install_mzbsuite:

.. code-block:: bash

    pip install -e .

the ``-e`` flag will install the package in editable mode, so that you can make changes to the functions in ``mzbsuite`` and they will be reflected in your environment. 

.. admonition:: \ \ 
   
   You can check whether ``mzbsuite`` has been installed by running: 

   .. code-block:: bash
    
    conda list -n mzbsuite

   and check that ``mzbsuite`` appears in the  list. 

If there are no errors then you're all set up and can start using the modules. 