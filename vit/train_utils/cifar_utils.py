from datasets import DatasetDict
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

def get_cifar_label_dicts(cifar_dataset:DatasetDict, label_type:str):
    """
    label_type: str 'coarse' or 'fine'
    returns
        label2id: dict
        id2label: dict
    """
    assert label_type in ("coarse","fine"), "label type must be 'coarse' or 'fine'"
    # Get label mappings
    labels = cifar_dataset["train"].features[f"{label_type}_label"].names

    label2id = {str(label): str(i) for i, label in enumerate(labels)}
    id2label = {str(i): str(label) for i, label in enumerate(labels)}

    return label2id, id2label

def make_cifar_dataloaders(dataset:DatasetDict, batch_size=1):
    # Load the CIFAR dataset
    cifar_train = dataset["train"]
    cifar_test = dataset["test"]

    # Define transforms
    _transforms = Compose([ToTensor()])  # Add more transforms as needed to prevent overfitting

    def preprocess_transforms(data_examples):
        data_examples["pixel_values"] = [_transforms(img) for img in data_examples["img"]]
        del data_examples["img"]
        return data_examples

    # Apply transformations
    cifar_train = cifar_train.with_transform(preprocess_transforms)
    cifar_test = cifar_test.with_transform(preprocess_transforms)

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=cifar_train,
        batch_size=batch_size,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=cifar_test,
        batch_size=batch_size,
        shuffle=True
    )

    return train_dataloader, test_dataloader
