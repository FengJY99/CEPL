from exp.cross_modal_engine.datasets.oxford_pets import OxfordPets
from exp.cross_modal_engine.datasets.oxford_flowers import OxfordFlowers
from exp.cross_modal_engine.datasets.fgvc_aircraft import FGVCAircraft
from exp.cross_modal_engine.datasets.dtd import DescribableTextures
from exp.cross_modal_engine.datasets.eurosat import EuroSAT
from exp.cross_modal_engine.datasets.stanford_cars import StanfordCars
from exp.cross_modal_engine.datasets.food101 import Food101
from exp.cross_modal_engine.datasets.sun397 import SUN397
from exp.cross_modal_engine.datasets.caltech101 import Caltech101
from exp.cross_modal_engine.datasets.ucf101 import UCF101
from exp.cross_modal_engine.datasets.imagenet import ImageNet
from exp.cross_modal_engine.datasets.imagenetv2 import ImageNetV2
from exp.cross_modal_engine.datasets.imagenet_sketch import ImageNetSketch
from exp.cross_modal_engine.datasets.imagenet_a import ImageNetA
from exp.cross_modal_engine.datasets.imagenet_r import ImageNetR


dataset_classes = {
    "oxford_pets": OxfordPets,
    "oxford_flowers": OxfordFlowers,
    "fgvc_aircraft": FGVCAircraft,
    "dtd": DescribableTextures,
    "eurosat": EuroSAT,
    "stanford_cars": StanfordCars,
    "food101": Food101,
    "sun397": SUN397,
    "caltech101": Caltech101,
    "ucf101": UCF101,
    "imagenet": ImageNet,
    "imagenetv2": ImageNetV2,
    "imagenet_sketch": ImageNetSketch,
    "imagenet_a": ImageNetA,
    "imagenet_r": ImageNetR,
}

dataset_name = {
    "oxford_pets": "OxfordPets",
    "oxford_flowers": "OxfordFlowers",
    "fgvc_aircraft": "FGVCAircraft",
    "dtd": "DescribableTextures",
    "eurosat": "EuroSAT",
    "stanford_cars": "StanfordCars",
    "food101": "Food101",
    "sun397": "SUN397",
    "caltech101": "Caltech101",
    "ucf101": "UCF101",
    "imagenet": "ImageNet",
}