"""
Base functionality and interfaces for processing domain-specific graph representation into the GraphDict
representations and visualizations which are used throughout this package.
"""
import re
import inspect
import json
import typing as t

import click
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import visual_graph_datasets.typing as tv
from visual_graph_datasets.data import NumericJsonEncoder
from visual_graph_datasets.util import TEMPLATE_ENV
from visual_graph_datasets.util import JSON_LIST, JSON_DICT, JSON_ARRAY
from visual_graph_datasets.util import sanitize_indents


def identity(value: t.Any) -> t.Any:
    return value


def list_identity(value: t.Any, dtype: type = float) -> t.Any:
    return [dtype(value)]


class ProcessingError(Exception):
    """
    An exceptions specifically to be used when there is a problem with the processing process.
    """
    pass


class ProcessingBase(click.MultiCommand):
    """
    Abstract base class for the implementation of the processing of a domain-specific graph representation
    into the generic GraphDict structure and visualizations which are used throughout this package.

    **Motivation**

    Most application domains have their own graph representation. An example for this would be SMILES strings
    which provide short and human-readable representations of molecular graphs or netlists in the case of
    electrical networks. The problem is that to apply these datasets to graph neural networks they always
    have to be preprocessed in some way to create a representation where nodes and edges are annotated with
    numeric feature vectors.

    A design choice of this package is that this pre-processing should be a property of the dataset itself.
    So essentially each dataset should be shipped in a format which contains all the GNN-ready graph
    representations already. This is done in an attempt to make results for the GNNs themselves more
    comparable by providing the exact same starting point for each of them.

    However, this opens another problem: If the preprocessing is done for the dataset a-priori, how do we
    format new and unknown elements into the same format? This would be necessary to really use the
    trained networks.

    That is the problem which this class aims to address: It provides a single place where the conversion of
    a domain specific graph representation to (1) a GNN-ready representation (2) a canonical representation
    can be created. More importantly this class implements a bit of magic: The implementation of the
    preprocessing functionality done in a subclass can automatically be converted into a standalone python
    module, which also implements the same functionality. That python module can be imported, but it can
    also directly be used as a command line application, making the processing functionality programming
    language independent as well.

    **Automatic CLI Generation**

    This class implements the functionality to generate a command line interface automatically
    from the simple python implementations of the processing procedure. At the core level, this process is
    done for the following three methods:

    - ``process``: Should turn a domain-specific graph representation into a GraphDict
    - ``visualize``: Should create a visualization of the graph given by the domain-specific represen.
    - ``create``: Should create the visual graph dataset file representation of a graph, which consists
      of the metadata JSON file and a PNG visualization.

    During the construction of a class instance, the corresponding click.Command objects are automatically
    derived from these methods, which will be available over the command line with the exact same names.

    To achieve this, the methods are wrapped in all the necessary click decorators and added as additional
    properties of the object instance, which are later detected as valid command implementation for the
    command line interface. The methods are parsed using the ``inspect`` module to extract and transfer as
    much information as possible to their command line counterparts.

    - Each positional argument of the method is converted into a position "argument" of the command as well
    - Each argument with a default value is instead converted into a command line "option" of the same name
    - The types annotations of each method argument are detected and converted into click.ParamType
      instance, if possible. This means that the string input values are automatically cast to the correct
      python types.
    - The docstring is parsed and the main description of the method is used as the --help text of the
      command
    - The individual :param: descriptions in the docstring are also detected and added as the descriptions
      of the corresponding command line "options"
    """

    # This string will be used as the main content of the description when using the --help text directly
    # on the top level command line interface, which is represented by this class.
    description: str = ''

    # This is a list of method names of this class.
    # This list determines which of the methods of this class will AUTOMATICALLY be constructed into CLI
    # commands upon the construction of an instance of this class.
    # For more information about how a method is dynamically turned into a Command see the method
    # "construct_command"
    DEFAULT_COMMANDS: t.List[str] = [
        'process',
        'visualize',
        'create',
    ]

    # This dictionary provides a mapping of "normal" python types to click.ParamTypes which are used in the
    # command line library click.
    # During the process of dynamically construction a click.Command from a method, the parameters of that
    # method and most importantly their type annotations are analyzed and those python type annotations are
    # then mapped into click parameter type instances using this dictionary.
    # This process is important because in the case of click, the type definitions are not just syntactic
    # sugar, but they also contain the functionality to automatically cast the string input values received
    # from the command line into the corresponding types!
    TYPE_CONVERSION_DICT = {
        int: click.INT,
        str: click.STRING,
        bool: click.BOOL,
        list: JSON_LIST,
        t.List: JSON_LIST,
        dict: JSON_DICT,
        t.Dict: JSON_DICT,
        np.ndarray: JSON_ARRAY,
    }

    def __init__(self, *args, **kwargs):
        super(ProcessingBase, self).__init__(*args, help=self.description, **kwargs)

        # This method implements the process of dynamic construction of the click.Commands from several of
        # the methods of this class
        self.scaffold()
        try:
            self.__code__ = inspect.getsource(self.__class__)
        except (AttributeError, TypeError):
            pass

    # -- to be implemented --
    # All of these methods have to be implemented by subclasses. They need to contain the domain specific
    # implementations if how to actually process domain-specific graph representations into the graph
    # dict / visualizations.

    def process(self,
                value: tv.DomainRepr,
                *args,
                additional_graph_data: dict = {},
                **kwargs
                ) -> tv.GraphDict:
        """
        This method needs to implement the transformation of a domain-specific graph representation into a
        GraphDict representation, as it is used throughout this package. At the core, this means that
        the method should return a dict which contains the 4 keys:

        - node_indices: A list of integer indices for all the nodes
        - node_attributes: A list of node feature vectors (lists), where each one defines the dataset
            specific feature values for each node
        - edge_indices: A list of tuples. Each tuple defines a directed edge in the graph and thus consists
            of two node indices.
        - edge_attributes: A list of edge feature vectors (lists) for each edge defined in the edge indices
            list.

        The resulting graph dictionary representation may contain more custom fields to support custom
        features.

        But beware that it has to be possible to JSON encode the dictionary! This means that there cannot be
        any "fancy" additional fields such as callback functions or custom objects, only native data types!

        :param value: The domain-specific representation of the graph to be converted
        :param additional_graph_data: A dictionary with string keys and arbitrary (json encodable) values
            that will be added as additional attributes to the graph dict

        :returns: The graph dict representation of the given element
        """
        raise NotImplemented()

    def visualize_as_figure(self,
                            value: tv.DomainRepr,
                            *args,
                            width: int,
                            height: int,
                            **kwargs
                            ) -> t.Tuple[plt.Figure, np.ndarray]:
        """
        This method needs to implement the transformation of a domain-specific graph representation into a
        visualization of the corresponding graph.
        This method should return a tuple consisting of two values:
        - The visualization as a plt.Figure object
        - A numpy array (V,2) containing the 2D coordinates of the nodes within that visualization

        :param value: The domain-specific representation of the graph to be converted
        :param width: The width of the image in pixels
        :param height: The height of the image in pixels

        :returns: Tuple (fig, node_positions)
        """
        raise NotImplemented()

    def visualize(self,
                  value: tv.DomainRepr,
                  width: int,
                  height: int,
                  **kwargs
                  ) -> np.ndarray:
        """
        This method needs to implement the transformation of a domain-specific graph representation into a
        visualization of the corresponding graph. This method in particular should return that visualization
        as a numpy rgb image array with the dimensions (width, height, 3).

        This choice of the array return type was to support it being possible to print the image to the
        command line as a string (JSON encoded).
        However, it is encouraged to also define an additional method which returns the visualization in a
        format that is better suited for direct code access, such as a matplotlib Figure object.

        :param value: The domain-specific representation of the graph to be converted
        :param width: The width of the image in pixels
        :param height: The height of the image in pixels

        :returns: A numpy array (width, height, 3) which represents the visualization RGB image
        """
        raise NotImplemented()

    def create(self,
               value: tv.DomainRepr,
               index: str,
               width: int,
               height: int,
               output_path: str,
               additional_graph_data: dict,
               additional_metadata: dict,
               *args,
               **kwargs,
               ) -> tv.MetadataDict:
        """
        This method needs to implement the creation of a visual graph dataset file representation of a graph,
        given its domain specific representation. This method is not supposed to return anything. Instead,
        it is supposed to directly create the necessary files.

        A visual graph dataset element is represented by two files:

        - A metadata JSON files, which contains the full GraphDict representation of the corresponding
          graph as well as all additional metadata, such as the index within the dataset, name etc.
        - A PNG image, which shows a visualization of the corresponding graph, which can then
          be used in the future to visualize explanations with.

        This method should return the metadata dictionary for that element.

        :param value: The domain-specific representation of the graph to be converted
        :param index: The index of the newly created elements in the larger dataset. This value will also
            be used as the file name for the two created files.
        :param width: The width of the image in pixels
        :param height: The height of the image in pixels
        :param output_path: The path of an existing FOLDER into which the files will be saved
        :param additional_graph_data: A dictionary with string keys and arbitrary (json encodable) values
            that will be added as additional attributes to the graph dict
        :param additional_metadata: A dictionary with string keys and arbitrary (json encodable) values
            that will be added as additional attributes to the metadata dict

        :returns: metadata dict
        """
        raise NotImplemented()

    def get_description_map(self) -> dict:
        """
        This method should return a dictionary, which contains descriptions of the results of the processing
        functionality implemented in the class.

        The dictionary should contain at least two keys:

        - node_attributes: A dictionary itself, whose keys are the indices of any node feature vector and
          the corresponding value is a natural language description of what kind of node property that value
          represents
        - edge_attributes: Also a dictionary, the same thing for the edge feature vectors.
        """
        raise NotImplemented()

    def __imports__(self):
        """
        One way in which this class will be used is by copying its entire source code into a separate
        python module, which will then be shipped with each visual graph dataset as a standalone input
        processing functionality.

        All the code of a class can easily be extracted and copied into a template using the "inspect"
        module, but it may need to use external imports which are not present in the template by default.
        This is the reason for this method.

        Within this method all necessary imports for the class to work properly should be defined. The code
        in this method will then be extracted and added to the top of the templated module in the imports
        section.
        """
        pass

    # -- dynamic command generation magic --

    def scaffold(self):
        """
        This method turns all the methods of this class which are listed in the DEFAULT_COMMANDS list by
        name into corresponding click.Command instances automatically and attaches them as properties to
        this instance.

        :returns: None
        """
        for command_name in self.DEFAULT_COMMANDS:
            command_method_name = f'{command_name}_command'

            # ONLY if the class does not already have a method with the given name do we actually
            # dynamically create one!
            # This is because the users should be able to define their own custom implementation of the
            # cli method if they want some extra custom behavior. The automatic scaffolding based on the
            # main functions is just a convenience feature.
            if not hasattr(self, command_method_name):
                method = getattr(self, command_name)

                # Given a name and a callable object, this method will dynamically construct a click command
                # object which implements that method as a CLI command.
                command = self.construct_command(command_name, method)
                # This command object is then added as a class attribute, as that means that it can be
                # discovered as an actual CLI command later.
                setattr(self, command_method_name, command)

    @classmethod
    def construct_command(cls, name: str, method: t.Callable) -> click.Command:
        """
        This method constructs a click.Command instance, given the ``name`` for the command and the callable
        method instance ``method`` which is to act as the command implementation and base template.

        The resulting command will use the functionality implemented in the given method when it is executed.

        :param name: the string name which the resulting command is supposed to go by
        :param method: A method callable instance of this class, which should be turned into the command.

        :returns: A command instance
        """
        # This function here will serve as the base template for the command object. This will now
        # progressively be wrapped in decorators to create the click command object which resembles the
        # given method.
        def command(**kwargs):
            value = method(**kwargs)

            # At the end of the
            if isinstance(value, (dict, list, np.ndarray)):
                click.echo(json.dumps(value, cls=NumericJsonEncoder))

        # We also need to transfer the doc string, because that doc string will later be displayed
        # as the help text in the command line.
        body, param_description_map = cls.parse_docstring(method.__doc__)
        command.__doc__ = body

        # In the first step we will use "inspect" to dynamically derive as many properties from the given
        # method as possible. This for example includes what kinds of arguments the method accepts and
        # also what the help text is.

        # ~ arguments and options
        # First of all we iterate through all the arguments of the method and correspondingly add the click
        # argument decorators such that the command has the same arguments as the method.
        arg_spec = inspect.getfullargspec(method)
        signature = inspect.signature(method)

        # We reverse the order of this list here because stacking the decorators on top of each other will
        # already reverse the order so by reversing the order here again we actually get the correct order
        # of the arguments as they appear in the method as well.
        for arg_name in reversed(arg_spec.args):
            # Obviously "self" is not actually an argument of the command but just an artifact of the method
            # and thus we ignore it here.
            if arg_name not in ['self']:
                arg = signature.parameters[arg_name]

                # By default, every argument will be assumed to be string. But here we try to learn more
                # from the type annotation of the argument. If a type annotation exists and it is a
                # simple type that we know (like int for example) we can get the corresponding click Type
                # object from the conversion dict and use that instead.
                # This is actually quite important because click performs automatic type casting of the
                # string cli inputs based on those annotations!
                arg_type = click.STRING
                if arg.annotation in cls.TYPE_CONVERSION_DICT:
                    # 22.03.2023 - If the annotation is directly an instance of click.ParamType we
                    # obviously want to use that. Also fixed a bug, where I didn't check if the type is
                    # even defined in the map, which could cause KeyError for exotic annotations.
                    if isinstance(arg_type, click.ParamType):
                        arg_type = arg.annotation
                    elif arg_type in cls.TYPE_CONVERSION_DICT:
                        arg_type = cls.TYPE_CONVERSION_DICT[arg.annotation]
                    else:
                        arg_type = click.STRING

                # Only if the argument does not have a default parameter we actually consider it as a
                # positional argument in the CLI context! If the argument has a default value it rather
                # resembles a CLI option instead.
                if arg.default == inspect.Parameter.empty:
                    # Argument
                    command = click.argument(
                        arg_name,
                        type=arg_type
                    )(command)
                else:
                    # Option
                    help_string = (f'{param_description_map.get(arg_name, "")}'
                                   f'type: {str(arg.annotation)} '
                                   f'default: {arg.default}')
                    command = click.option(
                        f'--{arg_name}',
                        type=arg_type,
                        default=arg.default,
                        help=help_string
                    )(command)

        # Finally, in the last layer we need to wrap the function in the actual click Command decorator so
        # that it will be recognized as such.
        command = click.command(name)(command)

        return command

    def get_commands_dict(self) -> t.Dict[str, click.Command]:
        """
        Returns a dict which defines all the supported commands dynamically. The keys of the dict are the
        string names of the commands and the values are the actual click.Command object instance which
        implement the functionality of those commands.

        :returns: dict
        """
        command_dict = {}
        # Essentially we want to dynamically detect all the commands which are implemented for this class
        # We do this by iterating through all the properties and whenever we encounter a click.Command
        # instance we add that to the dict, because these are going to be the methods which were
        # decorated with the click.command() decorator.
        for name in dir(self):
            attribute = getattr(self, name)
            if isinstance(attribute, click.Command):
                # Each click.Command has a "name" attribute here which we use as the command name in the
                # dict as well and NOT the method name. The method name can be different
                name = attribute.name
                command_dict[name] = attribute

        return command_dict

    def list_commands(self, ctx):
        """
        This method returns a list of strings, where each string is the name of a command which is supported
        by the command line interface.

        This method is required implementation for the click.MultiCommand base class

        :param ctx: The click context object

        :returns: list of strings
        """
        # The "get_commands_dict" method returns a dictionary, whose keys are the string names of supported
        # commands and the values are the actual click.Command objects which wrap the corresponding method
        # objects that implement the command functionality.
        command_dict = self.get_commands_dict()
        return list(command_dict.keys())

    def get_command(self, ctx, cmd_name):
        """
        Given the string name of a command returns the corresponding click.Command object.

        This method is required implementation for the click.MultiCommand base class

        :param ctx: The click context object
        :param cmd_name: The string name of the command

        :returns: click.Command
        """
        # The "get_commands_dict" method returns a dictionary, whose keys are the string names of supported
        # commands and the values are the actual click.Command objects which wrap the corresponding method
        # objects that implement the command functionality.
        command_dict = self.get_commands_dict()
        if cmd_name in command_dict:
            return command_dict[cmd_name]

    # -- Utils --

    @classmethod
    def save_figure(cls, fig, path: str):
        fig.savefig(path)

    @classmethod
    def save_metadata(cls, data: dict, path: str):
        with open(path, mode='w') as file:
            content = json.dumps(data, cls=NumericJsonEncoder)
            file.write(content)

    def array_from_figure(self,
                          figure: plt.Figure,
                          width: int,
                          height: int
                          ) -> np.ndarray:
        """
        Given a matplotlib ``figure`` Figure instance, this method will turn that into a numpy rgb image
        array of the dimensions (``width``, ``height``, 3).

        This is a functionality which is often needed for the subclass implementations as it makes more
        sense to implement the visualization into a pyplot figure than to an array. This method can then
        be used to create the required array format for the "visualize" method from that figure.

        :returns: numpy array with 3 dimensions
        """
        # This section turns the image which is currently a matplotlib figure object into a numpy array
        # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
        canvas = FigureCanvas(figure)
        canvas.draw()
        array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        array = array.reshape((height, width, 3))

        return array

    @classmethod
    def parse_docstring(cls,
                        docstring: str
                        ) -> t.Tuple[str, dict]:
        """
        This method parses the docstring of a method and separates it into two parts, which it returns as a
        tuple. The first element is the string of the main body of the description. The second element is a
        dictionary, whose keys are the names of the parameters of the method and the values are the
        descriptions for those parameters given in the docstring (:param name: description).

        If a docstring does not contain the description with the :param: syntax for one of the parameters
        it will not appear in the dictionary!

        Also beware that this method only works with the spinx rst style of parameter descriptions! All
        other syntax styles are not handled by this method.

        :param docstring: The full docstring of a method which describes it's functionality as well as
            the sphinx syntax for the parameter descriptions.

        :returns: (str, dict)
        """
        # 22.03.2023 - We catch an error in the trivial case here where there exists no docstring
        if docstring is None or docstring == '':
            return '', {}

        # This expression essentially matches the entire docstring UNTIL the first occurrence of
        # a parameter specification.
        expr_main = re.compile(r'(?P<main>(\n|\s|.)+?)(?=(:param)|$)')
        matches: dict = expr_main.match(docstring).groupdict()
        content = matches['main']

        # The keys of this dictionary are going to be the names of the parameters and the values will be
        # the string descriptions (ONLY the descriptions)
        param_map = {}
        # This expression matches a parameter specification (:param name: description). Also supports multi
        # line descriptions.
        # holy moly, getting this regex right took me half an hour...
        expr_param = re.compile(r'(?P<param>:param\s(?P<name>.*):(?P<description>(.|\n)*?))(?=(:param)|$)')
        for m in expr_param.finditer(docstring):
            name = m['name']
            description = m['description']
            param_map[name] = description

        return content, param_map


def create_processing_module(processing: ProcessingBase,
                             template_name: str = 'process.py.j2'):
    """
    Given an instance ``processing`` of a subclass if ProcessingBase, this method will template the code
    string for an independent python module based on the functionality provided by the given ProcessingBase
    subclass.

    :param ProcessingBase processing: An instance of a subclass of the ProcessingBase base class
    :param str template_name: The string name of the template to be used from the template folder of the
        package

    :return: The code STRING for the module. This will still have to be written into a file
    """
    template = TEMPLATE_ENV.get_template(template_name)

    # The obvious thing we need to extract from the given instance for this purpose is the source code of
    # the class.
    class_name = processing.__class__.__name__
    if hasattr(processing, '__code__'):
        code = processing.__code__
    else:
        code = inspect.getsource(processing.__class__)

    # 05.05.23 - This was a bug where I did not account for the possibility that the base Processing
    # class code might not be defined at the top-level indent, so here we make sure to catch that
    # possibility and normalize the indent
    code = sanitize_indents(code)

    imports_method = getattr(processing, '__imports__')
    raw_imports_code = inspect.getsource(imports_method)
    # The problem with the code which we get from that is that it is the code for the *entire* method which
    # also includes the name of the method and the arguments and the identation. But we only want the code
    # itself.
    # To achieve this we will apply regular expression magic. This regex will extract the method name
    # definition line and the actual content of the method into two named match groups
    expr = re.compile(r'(?P<def>.*def .*\((.|\n)*\):)(?P<content>(\n|.)*)')
    matches = expr.match(raw_imports_code).groupdict()
    # To get rid of the identation of the body, we first count the indent level of the definition line and
    # then for the content we assume it is one more indent level (+4 spaces). Then we remove that many
    # leading characters from each line of the content to then finally get only the content
    indent = len(matches['def']) - len(matches['def'].lstrip(' ')) + 4
    imports_code = '\n'.join([line[indent:] for line in matches['content'].split('\n')])

    context = {
        'class_name':   class_name,
        'code':         code,
        'imports_code': imports_code
    }
    return template.render(context)
