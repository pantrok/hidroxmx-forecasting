"""Data-layer stubs: R2-backed loaders for the hidroxai-mx snapshot.

Modules to be filled in Milestone 2 (§12.2):

- ``streams.py`` — lazy Parquet readers for ``series_hidrometricas.parquet``,
  ``series_climatologicas.parquet`` and ``feature_table.parquet``, scanning only
  the partitions required by the current basin / year window.
- ``static.py`` — static attribute vector ``a_b`` per sub-basin (area, slope,
  hypsometry, drainage density, channel geometry, IDW climatology).
- ``windows.py`` — sliding-window ``torch.utils.data.IterableDataset`` that
  materialises tensors on the fly from the Parquet stream.
- ``splits.py`` — temporal (train 2010–2020 · val 2021–2022 · test 2023–2025)
  and spatial (PUB / PUR) folds; extreme-event stratification.
"""
