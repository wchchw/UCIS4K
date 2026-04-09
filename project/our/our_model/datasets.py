from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class ForegroundUCISInsSegDataset(CocoDataset):
    """Dataset for UCIS4K."""

    METAINFO = {
        'classes': ['foreground'],
        'palette': [(0, 0, 255)]
    }



