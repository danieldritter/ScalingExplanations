import os
from typing import Dict, List, Tuple

from overrides import overrides
import numpy as np
import datasets 
from .utils import annotations_from_jsonl, load_flattened_documents, Evidence


class ERASERConfig(datasets.BuilderConfig):

    def __init__(self, features, data_url, citation, url, label_classes=("False", "True"), **kwargs):
        """
        Args:
          features: `list[string]`, list of the features that will appear in the
            feature dict. Should not include "label".
          data_url: `string`, url to download the zip file from.
          citation: `string`, citation for the data set.
          url: `string`, url for information about the data set.
          label_classes: `list[string]`, the list of classes for the label if the
            label is present as a string. Non-string labels will be cast to either
            'False' or 'True'.
          **kwargs: keyword arguments forwarded to super.
        """

        super(ERASERConfig, self).__init__(version=datasets.Version("1.0.2"), **kwargs)
        self.features = features
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url 

# class ERASERDataset(datasets.GeneratorBasedBuilder):

#     BUILDER_CONFIGS = [
#         ERASERConfig(
#             name="multirc",
#             version=datasets.Version("1.0.0", ""),
#             description="multirc portion of ERASER benchmark",
#         ),
#         ERASERConfig(
#             name="cos-e",
#             version=datasets.Version("1.0.0", ""),
#             description="multirc portion of ERASER benchmark",
#         ),
#         ERASERConfig(
#             name="e-snli",
#             version=datasets.Version("1.0.0", ""),
#             description="multirc portion of ERASER benchmark",
#         )
#     ]

#     def _info(self):
#         features = {feature: datasets.Value("string") for feature in self.config.features}
#         if self.config.name.startswith("wsc"):
#             features["span1_index"] = datasets.Value("int32")
#             features["span2_index"] = datasets.Value("int32")
#         if self.config.name == "wic":
#             features["start1"] = datasets.Value("int32")
#             features["start2"] = datasets.Value("int32")
#             features["end1"] = datasets.Value("int32")
#             features["end2"] = datasets.Value("int32")
#         if self.config.name == "multirc":
#             features["idx"] = dict(
#                 {
#                     "paragraph": datasets.Value("int32"),
#                     "question": datasets.Value("int32"),
#                     "answer": datasets.Value("int32"),
#                 }
#             )
#         elif self.config.name == "record":
#             features["idx"] = dict(
#                 {
#                     "passage": datasets.Value("int32"),
#                     "query": datasets.Value("int32"),
#                 }
#             )
#         else:
#             features["idx"] = datasets.Value("int32")

#         if self.config.name == "record":
#             # Entities are the set of possible choices for the placeholder.
#             features["entities"] = datasets.features.Sequence(datasets.Value("string"))
#             # Answers are the subset of entities that are correct.
#             features["answers"] = datasets.features.Sequence(datasets.Value("string"))
#         else:
#             features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)

#         return datasets.DatasetInfo(
#             description=_GLUE_DESCRIPTION + self.config.description,
#             features=datasets.Features(features),
#             homepage=self.config.url,
#             citation=self.config.citation + "\n" + _SUPER_GLUE_CITATION,
#         )

