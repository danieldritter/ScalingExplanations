import os
from typing import Dict, List, Tuple

from overrides import overrides
import numpy as np
import datasets 
import json 
from .utils import load_documents_from_file, load_documents, Evidence, Annotation, load_jsonl


class ERASERConfig(datasets.BuilderConfig):

    def __init__(self, features, data_dir=None, label_classes=("False", "True"), **kwargs):
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
        self.data_dir = data_dir

class ERASERHFDataset(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        ERASERConfig(
            name="multirc",
            features=["annotation_id","query","document","evidences"],
            description="multirc portion of ERASER benchmark",
        ),
        ERASERConfig(
            name="cose",
            features=["annotation_id","query","document","evidences"],
            description="cose portion of ERASER benchmark",
            label_classes=["A","B","C","D","E"]
        ),
        ERASERConfig(
            name="esnli",
            features=["annotation_id","premise","hypothesis","evidences"],
            description="esnli portion of ERASER benchmark",
            label_classes=["entailment", "neutral", "contradiction"]
        ),
    ]

    def _info(self):
        features = {feature: datasets.Value("string") for feature in self.config.features}
        features["evidences"] = datasets.features.Sequence({"text":datasets.Value("string"), "docid":datasets.Value("string")})
        features["label"] = datasets.features.ClassLabel(names=self.config.label_classes)

        return datasets.DatasetInfo(
            description=self.config.description,
            features=datasets.Features(features)
        )
    
    def _split_generators(self, dl_manager):
        task_name = self.config.name 
        dl_dir = os.path.join(self.config.data_dir, task_name)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "train.jsonl"),
                    "split": datasets.Split.TRAIN,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "val.jsonl"),
                    "split": datasets.Split.VALIDATION,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_file": os.path.join(dl_dir, "test.jsonl"),
                    "split": datasets.Split.TEST,
                },
            ),
        ]

    def _generate_examples(self, data_file, split):
        if self.config.name == "esnli" or self.config.name == "cose":
            documents  = load_jsonl(os.path.join(os.path.join(self.config.data_dir, self.config.name), "docs.jsonl"))
            documents = {doc['docid']: doc['document'] for doc in documents}

        with open(data_file, 'r') as inf:
            for line in inf:
                example = {} 
                content = json.loads(line)
                ev_groups = []
                for ev_group in content['evidences']:
                    ev_group = tuple([Evidence(**ev) for ev in ev_group])
                    ev_groups.append(ev_group)
                content['evidences'] = frozenset(ev_groups)
                annotation = Annotation(**content)
                if self.config.name == "multirc":
                    doc_name = annotation.annotation_id.split(":")[0]
                    document = load_documents(os.path.join(self.config.data_dir,self.config.name), set([doc_name]))
                    example["document"] = list(document.items())[0][1]
                    example["query"] = annotation.query 
                    example["evidences"] = [{"text":item.text, "docid":item.docid } for item in annotation.all_evidences()]
                    example["annotation_id"] = annotation.annotation_id 
                    example["label"] = annotation.classification 
                elif self.config.name == "esnli":
                    premise = load_documents_from_file(documents, set([annotation.annotation_id + "_premise"]))
                    hypothesis = load_documents_from_file(documents, set([annotation.annotation_id + "_hypothesis"]))
                    example["premise"] = list(premise.items())[0][1]
                    example["hypothesis"] = list(hypothesis.items())[0][1]
                    example["evidences"] = [{"text":item.text, "docid":item.docid } for item in annotation.all_evidences()]
                    example["annotation_id"] = annotation.annotation_id 
                    example["label"] = annotation.classification 
                elif self.config.name == "cose":
                    doc = load_documents_from_file(documents,set([annotation.annotation_id]))
                    example["document"] = list(doc.items())[0][1]
                    example["query"] = annotation.query 
                    example["evidences"] = [{"text":item.text, "docid":item.docid } for item in annotation.all_evidences()]
                    example["annotation_id"] = annotation.annotation_id 
                    example["label"] = annotation.classification 
                yield annotation.annotation_id, example