import torchvision

def get_dataset_cifar(path):
    train_dataset = torchvision.datasets.CIFAR100(root=path, download=True, train=True)
    val_dataset = torchvision.datasets.CIFAR100(root=path, download=True, train=False)
    dataset = {"train": [], "valid": []}
    for img, lab in train_dataset:
        dataset["train"].append({"image": img, "label": lab})
    for img, lab in val_dataset:
        dataset["valid"].append({"image": img, "label": lab})
    return dataset
