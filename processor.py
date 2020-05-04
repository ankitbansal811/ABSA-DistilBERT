# coding=utf-8

"""Processors for different tasks."""

import csv
import os

import pandas as pd

import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_examples(self, data_dir, set_type):
        """Gets a collection of InputExample's for any given set.
        data_dir: str - data directory
        set_type: str - choices[train, test, dev]"""
        raise NotImplementedError


    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class Sentihood_QA_M_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(os.path.join(data_dir, "train_QA_M.tsv"),sep="\t").values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_QA_M.tsv"),sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(os.path.join(data_dir, "test_QA_M.tsv"),sep="\t").values
        return self._create_examples(test_data, "test")

    def get_examples(self, data_dir, set_type):
        """See base class."""
        file = '{}_QA_M.tsv'.format(set_type)
        data = pd.read_csv(os.path.join(data_dir, file), sep="\t").values
        return self._create_examples(data, set_type)

    def get_labels(self):
        """See base class."""
        return ['None', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
          #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            text_b = tokenization.convert_to_unicode(str(line[2]))
            label = tokenization.convert_to_unicode(str(line[3]))
            if i%2000==0:
                print(i, end='')
                print("guid=",guid, end='')
                print("text_a=",text_a, end='')
                print("text_b=",text_b, end='')
                print("label=",label)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
