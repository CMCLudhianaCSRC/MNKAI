"""Cheap sanity test — every public module imports cleanly."""


def test_all_modules_import():
    import mnkai  # noqa: F401
    from mnkai import losses  # noqa: F401
    from mnkai.data import datasets, manifest, splits, transforms  # noqa: F401
    from mnkai.evaluation import bootstrap, metrics, subgroups  # noqa: F401
    from mnkai.models import backbones  # noqa: F401
    from mnkai.training import ema, sampler  # noqa: F401
    from mnkai.utils import config, logging, seed  # noqa: F401
