Other scripts
#############

Here we briefly overview other convenience scripts provided for facilitate other tasks not directly covered in other parts of the project. Please note these are not wrapped in ``__main__`` functions, and should be run from an interactive session (e.g. Jupyter notebook), and that all path names are hardcoded and should be changed directly in the scripts. 

.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

Pre-processing filenames
************************
The script ``scripts/diverse_preprocessing/preprocess_filenames.py`` standardises the names of the files in the provided path ``main_root``, by changing capitalization to lowercase and changing spaces to ``_``. 
It then parses the parent directories names and standardises them in the same way, and removes trailing spaces from filenames. 
The last step removes all files ``_mask.jpg`` that could have been left from previous iterations of the pipeline if output folders were not separated from input folders. 

Unzip multi-part archives with Python
****************************************
It might not always be possible to use a GUI utility to uncompress archives, and commonly images can come in a compressed archive. The script ``scripts/diverse_preprocessing/unzip_manual_skeletons.py`` unzips a multi-part archive into a folder, using the library ``zipfile``. 
