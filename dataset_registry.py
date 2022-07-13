from dataset_wrappers.multinli import MultiNLIDataset
from dataset_wrappers.spurious_sst import SpuriousSSTDataset
from dataset_wrappers.sst import SSTDataset
from dataset_wrappers.spurious_sst import SpuriousSSTDataset
from dataset_wrappers.hans import HansDataset
from dataset_wrappers.eraser.eraser_wrapper import ERASERDataset
DATASETS = {
    "multinli": MultiNLIDataset,
    "sst":SSTDataset,
    "spurious_sst":SpuriousSSTDataset,
    "hans":HansDataset,
    "eraser":ERASERDataset
}