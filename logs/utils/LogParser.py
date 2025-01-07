import sys

from .logparser import Drain
import os
from .ErrorHandling import validate_log_file


class LogParser:
    def __init__(self, input_file_path, regex_list=[], is_save_parsed_file=False):
        self.input_dir = os.path.dirname(input_file_path)
        self.saved_dir = 'Drain_result/'
        self.parser = self.make_parser(regex_list)
        self.is_save_parsed_file = is_save_parsed_file

    def make_parser(self, regex_list=[], dataset_name="Android"):
        print("Make Parser using {}".format(dataset_name))
        if dataset_name == "Android":
            log_format = '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>'
            regex = [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'
                     ] + regex_list

            st = 0.2  # Similarity threshold
            depth = 6  # Depth of all leaf nodes

        return Drain.LogParser(log_format, indir=self.input_dir, outdir=self.saved_dir, depth=depth, st=st,
                               rex=regex)

    def drain_parse(self, log_file):
        """
        :param log_file:
        :return:
        """
        self.parser.logName = log_file
        # if not os.path.isfile(self.parser.get_structed_path()):
        self.parser.parse(log_file, self.is_save_parsed_file)

    def drain_parse_with_tf(self, log_file):
        """
        :param log_file:
        :return:
        """
        self.parser.logName = log_file
        # if not os.path.isfile(self.parser.get_structed_path()):
        self.parser.parse_with_tf(log_file, self.is_save_parsed_file)


def parse_log_file_(file_path: str, regex_list: list = []):
    """
    ログファイルをParseし、tf値を計算。LogParserを返す。
    ライブラリ内部でのみ使用。
    :param file_path:
    :param regex_list:
    :return: df_structed, df_templates, logparser
    """
    # Validate file path
    validate_log_file(file_path)

    logparser = LogParser(file_path, regex_list=regex_list)
    logparser.is_save_parsed_file = True

    # Parse Procedure and calc TF-Value
    logparser.drain_parse(os.path.basename(file_path))
    df_structed = logparser.parser.df_log
    df_templates = logparser.parser.df_templates

    return df_structed, df_templates, logparser


def parse_log_file(file_path: str):
    """
    ログファイルをParseし、tf値を計算。
    :param file_path:
    :return: df_structed, df_templates
    """
    # Validate file path
    validate_log_file(file_path)

    logparser = LogParser(file_path)
    logparser.is_save_parsed_file = True

    # Parse Procedure and calc TF-Value
    logparser.drain_parse_with_tf(os.path.basename(file_path))
    df_structed = logparser.parser.df_log
    df_templates = logparser.parser.df_templates

    del logparser
    return df_structed, df_templates


def parse_logs(logparser: LogParser, target_logs: list):
    logCluL = logparser.parser.parse_raw_log(target_logs)
    template_strs = []
    for logClust in logCluL:
        template_str = " ".join(logClust.logTemplate)
        template_strs.append(template_str)

    return template_strs
