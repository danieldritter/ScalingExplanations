from dataset_wrappers.multinli import MultiNLIDataset
from dataset_wrappers.spurious_sst import SpuriousSSTDataset
from dataset_wrappers.sst import SSTDataset
from dataset_wrappers.spurious_sst import SpuriousSSTDataset
from dataset_wrappers.hans import HansDataset
DATASETS = {
    "multinli": MultiNLIDataset,
    "sst":SSTDataset,
    "spurious_sst":SpuriousSSTDataset,
    "hans":HansDataset
}