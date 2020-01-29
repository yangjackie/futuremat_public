from abc import ABCMeta, abstractmethod

class FileReader(object):
    """
    A generic file reader for reading text-based files on disk. A file reader can be initialized with either
    one of the following two parameters:

    :param str input_location: The path pointing to the file to be read.
    :param List file_content: A comma separated list of file-content for the file to be read in. This option
    is useful for testing purpose.

    Under the hood, the initializer will always return the file content extracted from the file (if a
    file location is given) to be parsed for specific parser.
    """
    def __init__(self, input_location=None, file_content=None):
        try:
            assert input_location is not None
            self.input_location = input_location
            self.__initialize_from_input_location()
        except AssertionError:
            assert file_content is not None
            self.__initialize_from_file_content(file_content)

    def __initialize_from_input_location(self):
        """
        Method to check if file specified at the path exists, if so, open the file, and
        extract the content of the file.
        """
        self.__initialize_from_file_content(open(self.input_location, 'r').read())

    def __initialize_from_file_content(self, file_content):
        """
        Take the file content, if it is a string, split into list of strings at the line breaks.
        Or if the content has been splitted, return the file content as the splitted list.

        :param file_content: the file content.
        :type file_content: str or List
        """
        if isinstance(file_content, str):
            self.file_content = file_content.split("\n")
        else:
            assert isinstance(file_content, list)
            self.file_content = file_content

    def read(self):
        """
        Generic method for parsing the file content.
        """
        pass


class FileWriter(object):
    """
    A generic file writer to write out the file content to a specific location on disc. A file reader must be
    initialised with the following parameters:

    :param str output_location: The path pointing to the file to be written out.
    :param List file_content: An object containing the content of the file to be written out.
    """
    def __init__(self,output_location,file_content):
        self.output_location = output_location
        self.file_content = file_content


class ObjectMapper(object):
    """
    Abstract class providing methods to map between domain models and contents stored in namedturples.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def map_to_object(self,content):
        """
        Method to map the contents stored in a namedturple to an object.

        :param content: A namedturple containing the contents read in from external data source.
        :return: A fully constructed CSPy domain model.
        """
        pass

    @abstractmethod
    def map_to_content(self,object):
        """
        Method to map the states of an object to a namedturple.

        :param object: A CSPy domain model
        :return: A namedturple containing the contents of the domain model, which can be written out to external
        data storage.
        """
        pass