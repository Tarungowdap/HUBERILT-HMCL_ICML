"""The 15-task sequence used by HMCL."""

RETRIEVAL_DATASETS = frozenset({"coco", "flickr30k"})

DATASET_SEQUENCE = (
    "aircraft,caltech101,cifar10,cifar100,clevr,coco,country211,dtd,"
    "eurosat,flickr30k,food101,mnist,flowers102,pcam,sst2"
)

ALL_DATASETS = tuple(name.strip() for name in DATASET_SEQUENCE.split(",") if name.strip())


def is_retrieval_dataset(name: str) -> bool:
    return name.lower() in RETRIEVAL_DATASETS
