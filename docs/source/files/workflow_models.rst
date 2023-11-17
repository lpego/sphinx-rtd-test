Workflow and Models
###################
Here we illustrate the main functionalities of the modules and information about the models. 
We also show the usage of ``workflows`` and its ``.sh`` files. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Introduction
============

This project contains 3 modules: 

#. Module :ref:`files/scripts/processing_scripts:Segmentation`: this module is mainly used as preprocessing steps for classification and measurements, but contains handy functionalities on its own. The raw input images are wide RGB photographs of many MZB organisms, which are hard to process on their own. A first step is to detect all the insects as independent and disjoint objects, and to crop them out of the original image. This is done with traditional image processing methods, and it is completely unsupervised. From this, we derive i) an RGB crop of the organism, with filename corresponding to the original image name plus clip ID and ii) a binary mask of the insect, filename corresponding to the original image name plus a clip ID. The binary mask can be used to compute the area of the insect (number of pixels) and RGB + clip can be used to compute local descriptors. 

#. Module :ref:`files/scripts/processing_scripts:Skeleton Prediction`: this module contains the code to measure the length and width of organisms. The user can chose two approaches: completely unsupervised or supervised. The first, unsupervised approach uses image filtering, mathematical morphology and a graph-traversal algorithm to come up with a measure approximating *only the length* of the organisms, made from a the mask obtained from the original images; performance is better for samples that are long and thin animals, slightly less accurate for complex shapes. The second approach to measure size is based on supervised Deep Learning (DL) models, which are trained based on manual annotations provided by a human expert of insects length (head to tail) and head width (width of head); the two predictions of the model correspond to the probability of each pixel to be along the "body" skeleton segments, or along the "head" skeleton segments. Postprocessing thins those predictions to a line, and the size of the sample is then approximated by the sum of the length of the body and the head, respectively. The scripts in this repo also allow users to finetune this DL model on their own data, and to use it for inference on new images. 

#. Module :ref:`files/scripts/processing_scripts:Classification`: this module contains the code to train and test a model to classify the organisms in MZB samples, according to a a set of predefined classes. In our experiments, image classes were specified in the filename, but this can be changed to a more flexible approach. We reclassify class names thanks to the :ref:`files/configuration:The taxonomy file`, which groups classes according to a taxonomic hierarchy. The data used to fine tune pretrained DL classifiers is then duplicated according to this hierarchical structure (each class in its own folder) in ``data/learning_sets/{project_name}/aggregated_learning_sets/``.

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Working with the project
========================

Project structure
~~~~~~~~~~~~~~~~~

- ``data/``: is meant to contain the raw image data, and the processed images (e.g. image clips, masks, skeletons, etc) that are derived from the raw data, as well as the taxonomy file (see :ref:`files/configuration:The taxonomy file`). 

    .. note:: 
        **TODO** Some CSV with derived measures and features are momentarily stored there but might not always be needed. We can maybe generate those conditionally by addition option in ``configs/global_configuration.yaml``.

- ``mzbsuite/``: contains the functions for each module; these are called in the scripts provided but can also be called from users' own scripts. 
- ``scripts/``: contains the scripts to run the different modules. Each module has its own folder, and each script is named after the module it implements. These modules can *be run in isolation* (i.e. each one can be run independently from the others from the command line). Arguments to these scripts are only the paths to the input and output folders, as well as the name of the project and model names (see also :ref:`files/workflow_models:Available models`). Users just need to specify the location of their own data and output folders on which to perform inference.  Some work might be needed to ensure that they can also be run interactively, but for this, it is better to duplicate those files in a ``notebooks/`` folder, and to run them interactively from there after modding them. 

    .. note:: 
        **TODO** Need to make so that renku workflow can track main inputs and outputs without making it too complex. 

- ``models/``: contains the pretrained models used in the different modules. The models are first downloaded from pytorch model zoo, then finetuned. The finetuned models are then saved in this folder.
- ``configs/``: contains the project configuration file, ``config.yaml``, which contains all the parameters for the different modules. This file contains all settings and hyperparameters, and it can be modified by the user to change the behavior of the scripts. The configuration files is always used as input to ``main`` scripts. 

    .. note:: 
        **TODO** Maybe good idea to create copies of this config, with per-experiment naming, or create a branch of the repo, etc. Make sure to version those also, with good commit names. 

- ``results/``: contains the results of the different modules. Each module has its own folder, and each script is named after the module it implements. It probably needs to be better structured and organized. 
- ``workflows/``: will contain the renku workflows, for now contains just an implementation of the serial pipeline in bash scripts. One for the inference pipeline, and two for finetuning the classification and size measurement models.

Workflow files
~~~~~~~~~~~~~~

In this folder, there are "workflow" files written in bash (``.sh``)that can be used to run the pipeline. Those files are nothing else that a chain of python commands implementing the flow of the processing pipeline. For instance, just run ``./workflows/run_finetune_skeletonization.sh`` to fine tune the skeletonization model. 

    .. note:: 
        **TODO**: transfer those bash scripts to renku workflows, so that the renku can track the dependencies and the inputs and outputs of each step, and generate the graph of the workflow.

Simple parameters in these files (e.g. input and output folders), together with the parameters in the configuration file, control the execution of the scripts. 

For example: 

.. code-block:: bash

    #!/bin/bash 

    ROOT_DIR="/data/shared/mzb-workflow"
    MODEL="efficientnet-b2-v0"
    LSET_FOLD=${ROOT_DIR}/data/learning_sets/project_portable_flume/aggregated_learning_sets

    python scripts/image_parsing/main_raw_to_clips.py \
        --input_dir=${ROOT_DIR}/data/raw/project_portable_flume \
        --output_dir=${ROOT_DIR}/data/derived/project_portable_flume/blobs/ \
        --save_full_mask_dir=${ROOT_DIR}/data/derived/project_portable_flume/full_image_masks \
        --config_file=${ROOT_DIR}/configs/configuration_flume_datasets.yaml
        -v

The extract above from ``workflows/run_pipeline.sh`` will run the script ``scripts/image_parsing/main_raw_to_clips.py``, passing it various global parameters: 

 - ``ROOT_DIR`` is the the root directory fo the project, this is important to anchor all the relative path references that modules make. 
 - ``MODEL`` is the name of the DL mode to be used, see list of models :ref:`files/workflow_models:Available models`
 - ``LSET_FOLD`` is the location where to copy the learning dataset prepared from the unparsed images

Then, the command ``python`` is invoked followed by the script to be executed, and the parameters required by that script, in this case: 

 - ``--input_dir`` where the raw images are stored
 - ``--output_dir`` where the outputs should be saved
 - ``--save_full_mask_dir`` where the full images with segmented masks should be saved
 - ``--config_file`` the location of the project configuration file (for more details see :ref:`files/configuration:Configuration`)
 - ``-v`` flag for verbose, printing out more details

Not all global parameters are used in the extract above, however they are necessary for other workflows; please also note that different scripts will require different input parameters, whereas the parameters contained in the configuration file are tied to the project and independent from the scripts. 

Interactive session
~~~~~~~~~~~~~~~~~~~

The full python environment is provided in ``environment.yml``. Users can use ``conda`` to compile the environment based on these specifications (see :ref:`files/installing:Manual Install`). 
    
    .. note:: 
        **TODO** We need to check if the docker image builds...

Instead of running the scripts using the workflows illustrated above, users can ruin directly the python files in an interactive session (e.g. Jupyter notebook). Note that in this case the script parameters must be supplied as variables in the interactive session and will be non-permanent. 

To launch a script in interactive session simply open it in a Jupyter notebook and set ``__name__ = '__main__'`` in your session. The code can now be run block-by-block and will print results to terminal; you can read more on ``__main__``  `here <https://docs.python.org/3/library/__main__.html>`_. 

Import as module
~~~~~~~~~~~~~~~~

Finally, users can import this repo as a package and make use of its functions in their own scripts. 
Make sure the package is installed (see :ref:`files/installing:Manual Install` in case of doubt), and simply use ``import mzbsuite`` in your script. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Available models
================
So far, these Depp Learning architectures are available for classification: 

- ``vgg``: VGG 16
- ``resnet18``: ResNet 18 layers
- ``resnet50``: ResNet 50 layers
- ``densenet161``: DenseNet 161 layers
- ``mobilenet``: MobileNet V2 
- ``efficientnet-b2``: EfficientNet B2
- ``efficientnet-b1``: EfficientNet B1
- ``vit16``: Vision Transformer 16 
- ``convnext-small``: ConvNext Small

The models are pre-trained on ImageNet and can be downloaded from the PyTorch model zoo. We use ``torchvision.models`` to load the models, and we pass ``weights={ModelName}_Weigths.IMAGENET1K_V1`` for the pre-trained weights. This can be changed depending on needs. 

Adding a new model
~~~~~~~~~~~~~~~~~~

In ``mzbsuite/utils.py`` you can either add a case to the function ``read_pretrained_model(architecture, n_class)`` or add a function returning a pytorch model. In general, the layers of these classifiers are all frozen and only the last fully connected layers are trained on the annotated data. This seemed to work in most of our cases, but can be changed in a simple way in the function. 

Logging your model's training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To be able to tell whether a model is learning properly and/or is overfitting, it's necessary to log its progress while training. We support two loggers for this: `Weights & Biases <https://docs.wandb.ai/>`_ and `TensorBoard <https://www.tensorflow.org/tensorboard>`_. 

To be able to use Weights & Biases you will need to create (free) account and install the necessary dependencies; refer to the documentation here: 

- Weights & Biases: `<https://wandb.ai/site/experiment-tracking>`_

After installing all requirements, run ``wandb login``.

For TensorBoard, please follow the installation instructions here: 

- TensorBoard: `<https://www.tensorflow.org/tensorboard/get_started>`_

You will also need to specify which logger to use in the ``model_logger`` parameter in the configuration file (see :ref:`files/configuration:Configuration`). 
