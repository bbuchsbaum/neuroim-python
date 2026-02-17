"""Source factory classes for lazy object construction from files.

Provides lazy-loading wrappers that defer reading neuroimaging data
from disk until explicitly requested via .load() or property access.
"""

from pathlib import Path
from typing import Union, Optional, Any

import numpy as np


class FileSource:
    """Base class for lazy file-backed neuroimaging objects.

    Parameters
    ----------
    file_path : str or Path
        Path to the neuroimaging file.
    meta_info : MetaInfo, optional
        Pre-loaded metadata. If None, metadata is read lazily.
    """

    def __init__(self, file_path: Union[str, Path], meta_info: Optional[Any] = None):
        self.file_path = Path(file_path)
        self._meta_info = meta_info
        self._loaded: Optional[Any] = None

    @property
    def meta(self) -> Any:
        """Return metadata without loading full data.

        Reads header information from the file if no MetaInfo was
        provided at construction time.
        """
        if self._meta_info is None:
            from .io import read_header
            self._meta_info = read_header(self.file_path)
        return self._meta_info

    def load(self) -> Any:
        """Materialize the underlying neuroimaging object.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement load()")

    def _ensure_loaded(self) -> Any:
        """Load on first access and cache the result."""
        if self._loaded is None:
            self._loaded = self.load()
        return self._loaded

    def __repr__(self) -> str:
        loaded = "loaded" if self._loaded is not None else "not loaded"
        return f"{type(self).__name__}(file_path='{self.file_path}', {loaded})"


class NeuroVolSource(FileSource):
    """Lazy loader for a single 3-D volume (DenseNeuroVol).

    Parameters
    ----------
    file_path : str or Path
        Path to the neuroimaging file.
    index : int, optional
        For 4-D files, which volume to extract (0-based). Default 0.
    meta_info : optional
        Pre-loaded metadata.
    """

    def __init__(self, file_path: Union[str, Path], index: int = 0,
                 meta_info: Optional[Any] = None):
        super().__init__(file_path, meta_info)
        self.index = index

    def load(self):
        """Read volume from disk.

        Returns
        -------
        DenseNeuroVol
        """
        from .io import read_vol
        self._loaded = read_vol(self.file_path, index=self.index)
        return self._loaded

    @property
    def data(self):
        """Access underlying data array (triggers load)."""
        return self._ensure_loaded().data

    @property
    def space(self):
        """Access NeuroSpace (triggers load)."""
        return self._ensure_loaded().space


class NeuroVecSource(FileSource):
    """Lazy loader for a 4-D vector (DenseNeuroVec).

    Parameters
    ----------
    file_path : str or Path
        Path to the 4-D neuroimaging file.
    indices : array-like, optional
        Volume indices to load (0-based). None loads all.
    meta_info : optional
        Pre-loaded metadata.
    """

    def __init__(self, file_path: Union[str, Path], indices=None,
                 meta_info: Optional[Any] = None):
        super().__init__(file_path, meta_info)
        self.indices = indices

    def load(self):
        """Read vector from disk.

        Returns
        -------
        DenseNeuroVec or SparseNeuroVec
        """
        from .io import read_vec
        self._loaded = read_vec(self.file_path, indices=self.indices)
        return self._loaded

    @property
    def data(self):
        """Access underlying data array (triggers load)."""
        return self._ensure_loaded().data

    @property
    def space(self):
        """Access NeuroSpace (triggers load)."""
        return self._ensure_loaded().space


class SparseNeuroVecSource(FileSource):
    """Lazy loader for a sparse 4-D vector (SparseNeuroVec).

    Parameters
    ----------
    file_path : str or Path
        Path to the 4-D neuroimaging file.
    mask : LogicalNeuroVol or array-like
        Binary mask defining the sparse region.
    indices : array-like, optional
        Volume indices to load (0-based). None loads all.
    meta_info : optional
        Pre-loaded metadata.
    """

    def __init__(self, file_path: Union[str, Path], mask,
                 indices=None, meta_info: Optional[Any] = None):
        super().__init__(file_path, meta_info)
        self.mask = mask
        self.indices = indices

    def load(self):
        """Read sparse vector from disk.

        Returns
        -------
        SparseNeuroVec
        """
        from .io import read_vec
        self._loaded = read_vec(self.file_path, indices=self.indices, mask=self.mask)
        return self._loaded

    @property
    def data(self):
        """Access underlying data array (triggers load)."""
        return self._ensure_loaded().data

    @property
    def space(self):
        """Access NeuroSpace (triggers load)."""
        return self._ensure_loaded().space


class MappedNeuroVecSource(FileSource):
    """Lazy loader for a MappedNeuroVec.

    Loads the underlying source vector from file, then wraps it
    with the given mapping function.

    Parameters
    ----------
    file_path : str or Path
        Path to the 4-D neuroimaging file.
    map_fun : callable
        Mapping function applied element-wise on read.
    inverse_fun : callable, optional
        Inverse of *map_fun* (enables writes).
    indices : array-like, optional
        Volume indices to load (0-based). None loads all.
    meta_info : optional
        Pre-loaded metadata.
    """

    def __init__(self, file_path: Union[str, Path], map_fun,
                 inverse_fun=None, indices=None,
                 meta_info: Optional[Any] = None):
        super().__init__(file_path, meta_info)
        self.map_fun = map_fun
        self.inverse_fun = inverse_fun
        self.indices = indices

    def load(self):
        """Read vector and wrap with mapping.

        Returns
        -------
        MappedNeuroVec
        """
        from .io import read_vec
        from .mapped_neuro_vec import MappedNeuroVec
        source = read_vec(self.file_path, indices=self.indices)
        self._loaded = MappedNeuroVec(source, self.map_fun, self.inverse_fun)
        return self._loaded

    @property
    def data(self):
        """Access mapped data array (triggers load)."""
        return self._ensure_loaded().data

    @property
    def space(self):
        """Access NeuroSpace (triggers load)."""
        return self._ensure_loaded().space
