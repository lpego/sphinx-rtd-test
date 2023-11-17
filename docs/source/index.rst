.. mzb-suite documentation master file, created by
   sphinx-quickstart on Fri Jun 30 09:14:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

================================================
MZB-WORFLOW: Macrozoobenthos data workflow suite 
================================================

This documentation introduces and illustrates some examples of the *Macrozoobenthos data processing workflow suite (mzb-suite)*. 

The MZB project
===============

:doc:`files/installing`
    How to install the project and ``mzb-suite`` package on your machine.

:doc:`files/workflow_models`
    Introduction to the project technical aspects and the models used.

:doc:`files/configuration`
    Explanation fo the configuration file and recommended parameter values. 


Examples
========

:doc:`files/examples/ex_intro`
    Overview of data processing abilities and limitations

:doc:`files/examples/ex_preprocessing`
    Example of data

:doc:`files/examples/segmentation`
    Example of extracting clips from large images

:doc:`files/examples/skeletonization_unsupervised`
    Example of extracting body length from organism clips

:doc:`files/examples/skeletonization_supervised_inference`
    Example of extracting body length and head width from organism clips

:doc:`files/examples/classification_inference`
    Example of automatically identifying taxa from organisms clips

:doc:`files/examples/classification_finetune`
    Example of retraining classification model


Processing scripts
==================

:doc:`files/scripts/processing_scripts`
    Detailed explanation for processing functions

:doc:`files/scripts/diverse_preprocessing`
    Details of other convenience scripts

.. Hidden TOCs

.. toctree::
    :caption: Project Documentation
    :maxdepth: 1
    :hidden:

    files/installing
    files/workflow_models
    files/configuration

.. toctree::
    :caption: Examples
    :maxdepth: 1
    :hidden:

    files/examples/ex_intro
    files/examples/ex_preprocessing
    files/examples/segmentation
    files/examples/skeletonization_unsupervised
    files/examples/skeletonization_supervised_inference
    files/examples/classification_inference
    files/examples/classification_finetune

.. toctree::
    :caption: Processing scripts
    :maxdepth: 1
    :hidden:

    files/scripts/processing_scripts
    files/scripts/diverse_preprocessing

.. toctree::
    :caption: mzb-suite Modules
    :maxdepth: 1 
    :hidden:

    files/modules/mzbsuite