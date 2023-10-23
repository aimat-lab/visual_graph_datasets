=========
CHANGELOG
=========

0.1.0 - 14.11.2022
------------------

* initial commit
* added ``rb-dual-motifs`` dataset
* added ``tadf`` dataset

0.2.0 - 15.11.2022
------------------

* Added module ``visual_graph_datasets.cli``
* Improved installation process. It's now possible to install in non-editable mode
* Added tests
* Added function ``get_dataset_path`` which returns the full dataset path given the string name of a
  dataset.

0.3.0 - 16.11.2022
------------------

* Added the dataset ``movie_reviews`` which is a natural language classification dataset which was
  converted into a graph dataset.
* Extended the function ``visual_graph_datasets.data.load_visual_graph_dataset`` to be able to load
  natural language text based graph datasets as well.

0.4.0 - 29.11.2022
------------------

Completely refactored the way in which datasets are managed.

* by default the datasets are now stored within a folder called ``.visual_graph_datasets/datasets``
  within the users home directory, but the datasets are no longer part of the repository itself.
  Instead, the datasets have to be downloaded from a remote file share provider first.
  CLI commands have been added to simplify this process. Assuming the remote provider is correctly
  configured and accessible, datasets can simply downloaded by name using the ``download`` CLI command.
* Added ``visual_graph_datasets.config`` which defines a config singleton class. By default this config
  class only returns default values, but a config file can be created at
  ``.visual_graph_datasets/config.yaml`` by using the ``config`` CLI command. Inside this config it is
  possible to change the remote file share provider and the dataset path.
* The CLI command ``list`` can be used to display all the available datasets in the remote file share.

0.5.0 - 02.11.2022
------------------

* Somewhat extended the ``AbstractFileShare`` interface to also include a method ``check_dataset`` which
  retrieves the files shares metadata and then checks if the provided dataset name is available from
  that file share location.
* Added the sub package ``visual_graph_datasets.generation`` which will contain all the functionality
  related to the generation of datasets.
* Added the module ``visual_graph_datasets.generation.graph`` and the class ``GraphGenerator`` which
  presents a generic solution for graph generation purposes.
* Added the sub package ``visual_graph_datasets.visualization`` which will contain all the functionality
  related to the visualization of various different kinds of graphs
* Added the module ``visual_graph_datasets.visualization.base``
* Added the module ``visual_graph_datasets.visualization.colors`` and functionality to visualize
  grayscale graphs which contain a single attribute that represents the grayscale value
* Added a ``experiments`` folder which will contain ``pyxomex`` experiments
* Added an experiment ``generate_mock.py`` which generates a simple mock dataset which will subsequently
  be used for testing purposes.
* Extended the dependencies

0.6.0 - 04.12.2022
------------------

* Added module ``visual_graph_datasets.visualization.importances`` which implements the visualization of
  importances on top of graph visualizations.
* Other small fixes, including a problem with the generation of the mock dataset
* Added ``imageio`` to dependencies

0.6.1 - 04.12.2022
------------------

* Default config now has the public nextcloud provider url

0.6.2 - 04.12.2022
------------------

* Fixed a bug with the ``list`` command which crashed due to non-existing terminal color specification

0.6.3 - 06.12.2022
------------------

* Finally finished the implementation of the ``bundle`` command.
* updated the rb_motifs dataset for the new structure and also recreated all the visualizations with a
  transparent background.
* Implemented the visualization of colored graphs

0.7.0 - 15.12.2022
------------------

* Changed the config file a bit: It is now possible to define as many custom file share providers as
  possible under the ``providers`` section. Each new provider however needs to have a unique name, which
  is then required to be supplied for the ``get_file_share`` function to actually construct the
  corresponding file share provider object instance.
* Added the package ``visual_graph_datasets.processing`` which contains functionality to process source
  datasets into visual graph datasets.
* Added experiment ``generate_molecule_dataset_from_csv`` which can be used to download the source CSV
  file for a molecule (SMILES based) dataset from the file share and then generate a visual graph dataset
  based on that.

0.7.1 - 15.12.2022
------------------

* Fixed a bug in the ``bundle`` command
* Added a module ``visual_graph_datasets.testing`` with testing utils.

0.7.2 - 16.12.2022
------------------

* Renamed ``TestingConfig`` to ``IsolatedConfig`` due to a warning in pytest test collection

0.8.0 - 30.12.2022
------------------

* Fixed a bug in ``experiments.generate_molecule_dataset_from_csv`` where faulty node positions were saved
  for the generated visualizations of the molecules
* Added the experiment ``experiments.generate_molecule_multitask_dataset_from_csv`` which generates a
  molecule based dataset for a multitask regression learning objective using multiple CSVs and merging
  them together.
* Fixed a bug in ``experiments.generate_molecule_multitask_dataset_from_csv`` where invalid molecules were
  causing problems down the line. These are being filtered now.
* Update ``README.md``
* Added a ``examples`` folder

0.9.0 - 30.12.2022
------------------

- Initial implementation of the "dataset metadata" feature: The basic idea is that special metadata files
  can be added to the various dataset folders optionally to provide useful information about them, such as a
  description, a version string, a changelog, information about the relevant tensor shapes etc... In the
  future the idea is to allow arbitrary metadata files which begin with a "." character. For now, the
  central ``.meta.yaml`` file has been implemented to hold the bulk of the textual metadata in a machine
  readable format.
- Added a main logger to the main config singleton, such that this can be used for the command line
  interface.
- Added the ``gather`` cli command which can be used to generate/update the metadata information for a
  single dataset folder. This will create an updated version of the ``.meta.yaml`` file within that folder.
- Changed the ``bundle`` command such that the metadata file is now always updated with the new dataset
  specific metadata, regardless of whether it exists or not. Additionally, custom fields added to that
  file which do not interfere with the automatically generated part now persist beyond individual bundle
  operations.
- Updated jinja template for ``list`` command to be more idiomatic and don't use logic within the template.
  Additionally extended it with more metadata information that is now available for datasets
- Switched to the new version of ``pycomex`` which introduces experiment inheritance.
- Started to implement more specific sub experiments using experiment inheritance.

**INTERFACE CHANGES**

- The central function ``load_visual_graph_dataset`` now has a backward-incompatible signature: The function
  still returns a tuple of two elements as before, but the first element of that tuple is now the metadata
  dict of the dataset as it was loaded by ``load_visual_graph_dataset_metadata``

0.9.1 - 27.02.2023
------------------

- changed dependencies to fit together with ``graph_attention_student``
- Added experiment to generate ``aggregators_binary`` dataset

0.10.0 - 20.03.2023
-------------------

Implemented the "preprocessing" feature. Currently a big problem with the visual graph datasets in general
is that they are essentially limited to the elements which they already contain. There is no easy way to
generate more input elements in the same general graph representation / format as the elements already in
a dataset. This is a problem if any model trained based on a VGD is supposed to be actually used on new
unseen data: It will be difficult to process a new molecule for example into the appropriate input tensor
format required to query the model.

The "preprocessing" feature addresses this problem. During the creation of each VGD a python module
"process.py" is automatically created from a template and saved into the VGD folder as well. It contains all
the necessary code needed to transform a domain specific implementation (such as a SMILES code for example)
into a new input element of that dataset, including the graph representation as well as the visualization.
This module can either be imported to use the functionality directly in python code. It also acts as a
command line application.

- Added the base class ``processing.base.ProcessingBase``. This class encapsulates the previously described
  pre-processing functionality. Classes inheriting from this automatically act as a command line interface
  as well.
    - Code for a standalone python module with the same processing functionality can be generated from an
      instance using the ``processing.base.create_processing_module`` function.
- Added the class ``processing.molecules.MoleculeProcessing``. This class provides a standard implementation
  for processing molecular graphs given as SMILES strings.
- Added unittests for base processing functionality
- Added unittests for molecule processing functionality

- Extended the function ``typing.assert_graph_dict`` to do some more in-depth checks for a valid graph dict
- Added module ``generation.color``. It implements utility functions which are needed specifically for the
  generation of color graph datasets.
- Added the experiment ``experiment.generate_rb_adv_motifs`` which generates the synthetic
  "red-blue adversarial motifs" classification dataset of color graphs.

0.10.1 - 24.03.2023
-------------------

- Changed the "config" cli command to also be usable without actually opening the editor. This can be used
  to silently create or overwrite a config file for example.

0.10.2 - 24.03.2023
-------------------

- Fixed a bug in ``utils.dynamic_import``
- Fixed a bug in ``data.load_visual_graph_element``

0.10.3 - 27.03.2023
-------------------

- Changed the version dependency for numpy

0.11.0 - 02.05.2023
-------------------

- Slightly changed the generation process of the "rb_adv_motifs" dataset.
- Added the class of experiments based on ``experiments.csv_sanchez_lengeling_dataset.py``, which convert
  the datasets from the paper into a single CSV file, which can then be further processed into a visual graph
  dataset.
- Added utility function ``util.edge_importances_from_node_importances`` to derive edge explanations from
  the node explanations in cases where they are not created.
- Started to move towards the new pycomex Functional API with the experiments
- Added more documentation to ``typing``

Model Interfaces and Mixins

- Added the ``visual_graph_datasets.models`` module which will contain all the code which is relevant for
  models that specifically work with visual graph datasets
- Added the ``models.PredictGraphMixin`` class, which is essentially an interface that can be implemented
  by a model class to signify that it supports the ``predict_graph`` method which can be used to query a
  model prediction directly based on a GraphDict object.

Examples

- Added a ``examples/README.rst``
- Added ``examples/01_explanation_pdf``

0.12.0 - 05.05.2023
-------------------

- Added a section about dataset conversion to the readme file
- Fixed a bug with the ``create_processing_module`` function where it did not work if the Processing class
  was not defined at the top-level indentation.
- Changed some dependency versions
- Moved some more experiment modules to the pycomex functional API

Important

- Made some changes to the ``BaseProcessing`` interface, which will be backwards incompatible
    - Mainly made the base interface more specific such as including "output_path" or "value" as concrete
      positional arguments to the various abstract methods instead of just specifying args and kwargs


0.12.1 - 20.05.2023
-------------------

- Added the ``Batched`` utility iterator class which will greatly simplify working in batches for
  predictions etc.
- Made some changes to the base molecule processing file
- Started moving more experiment modules to the new pycomex functional api
- Added an experiment module to process QM9 dataset into a visual graph dataset.

0.13.0 - 11.06.2023
-------------------

Additions to the ``processing.molecules`` module. Added various new molecular node/features based on
RDKit computations.

- Partial Gasteiger Charges of atoms
- Crippen LogP contributions of atoms
- Estate indices
- TPSA contributions
- LabuteASA contributions
- Changed the default experiment ``generate_molecule_dataset_from_csv.py`` to now use these additional
  atom/node features for the default Processing implementation.

Overhaul of the dataset writing and reading process. The main difference is that I added support for
*dataset chunking*. Previously a dataset would consist of a single folder which would directly contain all
the files for the individual dataset elements. For large datasets these folders would become very large and
thus inefficient for the filesystem to handle. With dataset chunking, the dataset can be split into multiple
sub folders that contain a max. number of elements each thus hopefully increasing the efficiency.

- Added ``data.DatasetReaderBase`` class, which contains the base implementation of reading a dataset from
  the persistent folder representation into the index_data_map. This class now supports the dataset
  chunking feature.
    - Added ``data.VisualGraphDatasetReader`` which implements this for the basic dataset format that
      represents each element as a JSON and PNG file.
- Added ``data.DatasetWriterBase`` class, which contains the base implementation of writing a dataset from
  a data structure representation into the folder. This class now supports the dataset chunking feature.
    - Added ``data.VisualGraphDatasetWriter`` which implements this for the basic dataset format where
      a metadata dict and a mpl Figure instance are turned into a JSON and PNG file.
- Changed the ``processing.molecule.MoleculeProcessing`` class to now also support a DatasetWriter instance
  as an optional argument to make use of the dataset chunking feature during the dataset creation process.

Introduction of COGILES (Color Graph Input Line Entry System) which is a method of specifying colored graphs
with a simple human-readable string syntax, which is strongly inspired by SMILES for molecular graphs.

- Added ``generate.colors.graph_from_cogiles``
- Added ``generate.colors.graph_to_cogiles``

Bugfixes

- I think I finally solved the performance issue in ``generate_molecule_dataset_from_csv.py``. Previously
  there was an issue where the avg write speed would rapidly decline for a large dataset, causing the
  process to take way too long. I *think* the problem was the matplotlib cache in the end
- Also changed ``visualize_graph_from_mol`` and made some optimizations there. It no longer relies on
  the creation of intermediate files and no temp dir either, which shaved of a few ms of computational time.


0.13.1 - 12.06.2023
-------------------

- Added the new module ``graph.py`` which will contain all GraphDict related utility functions in the future
  - Added a function to copy graph dicts
  - Added a function to create node adjecency matrices for graph dicts
  - Added a function to add graph edges
  - Added a function to remove graph edges

0.13.2 - 12.06.2023
-------------------

- Fixed a bug where ``ColorProcesing.create`` would not save the name or the domain representation 

0.13.3 - 12.06.2023
-------------------

- Fixed a bug where the COGILES decoding procedure produced graph dicts with "edge_attributes" arrays of 
  the incorrect data type and shape.

0.13.4 - 12.06.2023
-------------------

- Fixed a bug where the CogilesEncoder duplicated edges in some very weird edge cases!

0.13.5 - 23.10.2023
-------------------

- Added the experiment ``profile_molecule_processing.py`` to profile and plot the runtime of the different 
  process components that create a visual graph dataset element with the aim of identifying the source of the 
  runtime degradation bug.
