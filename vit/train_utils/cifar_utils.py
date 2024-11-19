from datasets import DatasetDict, Dataset, load_dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

_CIFAR_DS_TRANSFORMS = Compose([ToTensor()])

COARSE_ID2LABEL = {
    0 : "aquatic_mammals",
    1 : "fish",
    2 : "flowers",
    3 : "food_containers",
    4 : "fruit_and_vegetables",
    5 : "household_electrical_devices",
    6 : "household_furniture",
    7 : "insects",
    8 : "large_carnivores",
    9 : "Large_man-made_outdoor_things",
    10 : "large_natural_outdoor_scenes",
    11 : "large_omnivores_and_herbivores",
    12 : "medium_mammals",
    13 : "non-insect_invertebrates",
    14 : "people",
    15 : "reptiles",
    16 : "small_mammals",
    17 : "trees",
    18 : "vehicles_1",
    19 : "vehicles_2",
}

COARSE_LABEL2ID = {v: k for k, v in COARSE_ID2LABEL.items()}

FINE_ID2LABEL = {
    0 : "apple",
    1 : "aquarium_fish",
    2 : "baby",
    3 : "bear",
    4 : "beaver",
    5 : "bed",
    6 : "bee",
    7 : "beetle",
    8 : "bicycle",
    9 : "bottle",
    10 : "bowl",
    11 : "boy",
    12 : "bridge",
    13 : "bus",
    14 : "butterfly",
    15 : "camel",
    16 : "can",
    17 : "castle",
    18 : "caterpillar",
    19 : "cattle",
    20 : "chair",
    21 : "chimpanzee",
    22 : "clock",
    23 : "cloud",
    24 : "cockroach",
    25 : "couch",
    26 : "cra",
    27 : "crocodile",
    28 : "cup",
    29 : "dinosaur",
    30 : "dolphin",
    31 : "elephant",
    32 : "flatfish",
    33 : "forest",
    34 : "fox",
    35 : "girl",
    36 : "hamster",
    37 : "house",
    38 : "kangaroo",
    39 : "keyboard",
    40 : "lamp",
    41 : "lawn_mower",
    42 : "leopard",
    43 : "lion",
    44 : "lizard",
    45 : "lobster",
    46 : "man",
    47 : "maple_tree",
    48 : "motorcycle",
    49 : "mountain",
    50 : "mouse",
    51 : "mushroom",
    52 : "oak_tree",
    53 : "orange",
    54 : "orchid",
    55 : "otter",
    56 : "palm_tree",
    57 : "pear",
    58 : "pickup_truck",
    59 : "pine_tree",
    60 : "plain",
    61 : "plate",
    62 : "poppy",
    63 : "porcupine",
    64 : "possum",
    65 : "rabbit",
    66 : "raccoon",
    67 : "ray",
    68 : "road",
    69 : "rocket",
    70 : "rose",
    71 : "sea",
    72 : "seal",
    73 : "shark",
    74 : "shrew",
    75 : "skunk",
    76 : "skyscraper",
    77 : "snail",
    78 : "snake",
    79 : "spider",
    80 : "squirrel",
    81 : "streetcar",
    82 : "sunflower",
    83 : "sweet_pepper",
    84 : "table",
    85 : "tank",
    86 : "telephone",
    87 : "television",
    88 : "tiger",
    89 : "tractor",
    90 : "train",
    91 : "trout",
    92 : "tulip",
    93 : "turtle",
    94 : "wardrobe",
    95 : "whale",
    96 : "willow_tree",
    97 : "wolf",
    98 : "woman",
    99 : "worm",
}

FINE_LABEL2ID = {v: k for k, v in FINE_ID2LABEL.items()}

def get_label_dicts(label_type:str):
    """
    :param label_type: str 'coarse' or 'fine'
    
    :return id2label: dict[int] -> str
    :return label2id: dict[str] -> int
    """
    assert label_type in ("coarse","fine"), "label type must be 'coarse' or 'fine'"
    if label_type == "coarse":
        return COARSE_ID2LABEL, COARSE_LABEL2ID
    return FINE_ID2LABEL, FINE_LABEL2ID

def dataloaders_from_cfg(cfg:dict):
    batch_size = cfg["training"]["batch_size"]
    cifar_train:Dataset = load_dataset("uoft-cs/cifar100", split=cfg["cifar_dataset"]["train_split"])
    cifar_validation:Dataset = load_dataset("uoft-cs/cifar100", split=cfg["cifar_dataset"]["validation_split"])
    cifar_test:Dataset = load_dataset("uoft-cs/cifar100", split=cfg["cifar_dataset"]["test_split"])

    train_dataloader = _dataloader_from_dataset(cifar_train, batch_size=batch_size)
    validation_dataloader = _dataloader_from_dataset(cifar_validation, batch_size=batch_size)
    test_dataloader = _dataloader_from_dataset(cifar_test, batch_size=batch_size)

    return train_dataloader, validation_dataloader, test_dataloader


def _dataloader_from_dataset(dataset:Dataset, batch_size=1):
    def preprocess_transforms(data_examples):
        data_examples["pixel_values"] = [_CIFAR_DS_TRANSFORMS(img) for img in data_examples["img"]]
        del data_examples["img"]
        return data_examples

    # Apply transformations
    dataset = dataset.with_transform(preprocess_transforms)

    # Create dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader
