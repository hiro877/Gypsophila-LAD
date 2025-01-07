import os


"""
For LogFile Error
"""
class LogFileError(Exception):
    """Base class for exceptions in this module."""
    pass


class InvalidFilePathError(LogFileError):
    """Exception raised for invalid file paths."""

    def __init__(self, path):
        self.path = path
        self.message = f"Invalid file path: {self.path}. Please check if the file exists."
        super().__init__(self.message)


class InvalidFileFormatError(LogFileError):
    """Exception raised for invalid file formats."""

    def __init__(self, path):
        self.path = path
        self.message = f"Invalid file format: {self.path}. Expected a log or text file."
        super().__init__(self.message)


class FileReadError(LogFileError):
    """Exception raised for errors during file reading."""

    def __init__(self, path, reason):
        self.path = path
        self.reason = reason
        self.message = f"Failed to read file: {self.path}. Reason: {self.reason}"
        super().__init__(self.message)


def validate_file_path(file_path):
    """Validates the file path."""
    if not os.path.exists(file_path):
        raise InvalidFilePathError(file_path)


def validate_file_format(file_path):
    """Validates the file format."""
    if not file_path.endswith(('.log', '.txt')):
        raise InvalidFileFormatError(file_path)


def read_log_file(file_path):
    """Reads the log file and handles potential errors."""
    validate_file_path(file_path)
    validate_file_format(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
            return data
    except OSError as e:
        raise FileReadError(file_path, str(e))

def readlines_log_file(file_path):
    """Reads the log file and handles potential errors."""
    validate_file_path(file_path)
    validate_file_format(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.readlines()
            return data
    except OSError as e:
        raise FileReadError(file_path, str(e))

def validate_log_file(file_path):
    """Reads the log file and handles potential errors."""
    validate_file_path(file_path)
    validate_file_format(file_path)


# # Example usage
# try:
#     log_data = read_log_file('path/to/logfile.log')
#     # Proceed with analyzing log_data...
# except LogFileError as e:
#     print(e.message)
