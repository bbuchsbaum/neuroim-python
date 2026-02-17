import numpy as np
from typing import Tuple, Union, Optional, List
from .axis import (AxisSet, AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND,
                   NamedAxis, find_anatomy_3d, axis_names, axis_set)


class NeuroSpace:
    """Geometric specification of image space, mapping from grid coordinates to real coordinates.
    
    Direct translation of R's NeuroSpace class.
    
    Parameters
    ----------
    dim : tuple of int
        Dimensions of the image space
    spacing : tuple of float, optional
        Voxel dimensions (default: all 1.0)
    origin : tuple of float, optional  
        Origin coordinates (default: all 0.0)
    axes : AxisSet, optional
        Axis specification (default: determined from trans or standard axes)
    trans : ndarray, optional
        4x4 transformation matrix (default: created from spacing/origin)
        
    R Equivalent
    ------------
    neuroim2::NeuroSpace
    """
    
    def __init__(self, 
                 dim: Union[Tuple[int, ...], List[int], np.ndarray],
                 spacing: Optional[Union[Tuple[float, ...], List[float], np.ndarray]] = None,
                 origin: Optional[Union[Tuple[float, ...], List[float], np.ndarray]] = None,
                 axes: Optional[Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D]] = None,
                 trans: Optional[np.ndarray] = None):
        
        # Convert to numpy arrays
        self.dim = np.asarray(dim, dtype=int)
        
        # Validate dimensions
        if len(self.dim) == 0:
            raise ValueError("Dimensions cannot be empty")
        if np.any(self.dim <= 0):
            raise ValueError("All dimensions must be positive")
            
        ndim = len(self.dim)

        # Track whether user explicitly provided spacing/origin
        _spacing_provided = spacing is not None
        _origin_provided = origin is not None

        # Set defaults if not provided
        if spacing is None:
            spacing = np.ones(ndim)
        else:
            spacing = np.asarray(spacing, dtype=float)
            
        if origin is None:
            origin = np.zeros(ndim)
        else:
            origin = np.asarray(origin, dtype=float)
            
        self.spacing = spacing
        self.origin = origin
        
        # Validate spacing
        if np.any(self.spacing <= 0):
            raise ValueError("Spacing values must be positive")
        
        # Handle transformation matrix
        if trans is None:
            # Create transformation matrix from spacing and origin
            if ndim <= 3:
                self.trans = np.eye(4)
                self.trans[:ndim, :ndim] = np.diag(self.spacing)
                self.trans[:ndim, 3] = self.origin
            else:
                # For 4D+ spaces, use extended transformation matrix
                self.trans = np.eye(ndim + 1)
                self.trans[:ndim, :ndim] = np.diag(self.spacing)
                self.trans[:ndim, ndim] = self.origin
        else:
            self.trans = np.asarray(trans, dtype=float)
            # For 4D+ spaces, allow larger transformation matrices
            if ndim <= 3 and self.trans.shape != (4, 4):
                raise ValueError(f"trans must be 4x4 matrix for {ndim}D space, got {self.trans.shape}")
            elif ndim > 3 and self.trans.shape != (ndim + 1, ndim + 1):
                # Try to use 4x4 for first 3 dimensions
                if self.trans.shape == (4, 4):
                    # Extend to full size
                    full_trans = np.eye(ndim + 1)
                    full_trans[:3, :3] = self.trans[:3, :3]
                    full_trans[:3, -1] = self.trans[:3, 3]
                    # Additional dimensions get identity
                    for i in range(3, ndim):
                        full_trans[i, i] = self.spacing[i] if i < len(self.spacing) else 1.0
                    self.trans = full_trans
                else:
                    raise ValueError(f"trans must be {ndim+1}x{ndim+1} matrix for {ndim}D space, got {self.trans.shape}")

            # Extract spacing and origin from transformation matrix if not provided explicitly
            # Note: This matches R's behavior where trans takes precedence
            if trans is not None and not _spacing_provided:
                self.spacing = np.sqrt(np.sum(self.trans[:ndim, :ndim]**2, axis=0))
            if trans is not None and not _origin_provided:
                if ndim <= 3:
                    self.origin = self.trans[:ndim, 3]
                else:
                    self.origin = self.trans[:ndim, ndim]
        
        # Calculate inverse transformation
        try:
            self.inverse = np.linalg.inv(self.trans)
        except np.linalg.LinAlgError:
            # For 4D+ spaces, inverse might not be needed
            if ndim > 3:
                self.inverse = None
            else:
                raise ValueError("Transformation matrix is not invertible")
        
        # Handle axes
        # Convert list of axis names to AxisSet
        if isinstance(axes, (list, tuple)):
            from .axis import axis_set
            axes = axis_set(len(axes), list(axes))

        if axes is None:
            if ndim == 3 and trans is not None:
                # Use nearest anatomy like R does
                axes = self._nearest_anatomy()
            else:
                # Create default axes
                axes = self._default_axes(ndim)
        
        self.axes = axes
        
        # Validate dimensions match
        if self.axes.ndim != ndim:
            raise ValueError(f"Axes dimensionality ({self.axes.ndim}) must match space dimensionality ({ndim})")
    
    def _default_axes(self, ndim: int) -> Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D, AxisSetND]:
        """Create default axes for given dimensionality."""
        from .axis import axis_set
        names = ["x", "y", "z", "t", "v"][:ndim]
        if ndim > 5:
            names.extend([f"d{i}" for i in range(6, ndim + 1)])
        return axis_set(ndim, names)
    
    def _nearest_anatomy(self) -> AxisSet3D:
        """Determine anatomical orientation from transformation matrix.
        
        This determines the closest standard anatomical orientation
        (like RAS, LPI, etc.) based on the transformation matrix.
        """
        from .axis import find_anatomy_3d
        
        if self.trans is None or self.trans.shape[0] < 3:
            # Default to LPI if no transformation matrix
            return find_anatomy_3d("LPI")
        
        # Extract the 3x3 rotation/scale part of the transformation matrix
        # Make a COPY to avoid modifying self.trans in-place
        mat33 = self.trans[:3, :3].copy()
        
        # Normalize each column (axis direction)
        for i in range(3):
            col = mat33[:, i]
            norm = np.linalg.norm(col)
            if norm > 0:
                mat33[:, i] = col / norm
        
        # Find the best matching anatomical orientation
        best_score = -1
        best_orient = "LPI"
        
        # Check all possible anatomical orientations
        orientations = ["RAS", "LAS", "RPS", "LPS", "RAI", "LAI", "RPI", "LPI",
                       "RSA", "LSA", "RSP", "LSP", "RIA", "LIA", "RIP", "LIP",
                       "ARS", "ALS", "PRS", "PLS", "ARI", "ALI", "PRI", "PLI",
                       "ASR", "ASL", "PSR", "PSL", "AIR", "AIL", "PIR", "PIL",
                       "SRA", "SLA", "SRP", "SLP", "IRA", "ILA", "IRP", "ILP",
                       "SAR", "SAL", "SPR", "SPL", "IAR", "IAL", "IPR", "IPL"]
        
        for orient in orientations:
            try:
                anat = find_anatomy_3d(orient)
                # Get the permutation matrix for this orientation
                anat_mat = np.zeros((3, 3))
                axis_attrs = ['i', 'j', 'k']
                for i in range(3):
                    axis = getattr(anat, axis_attrs[i])
                    if isinstance(axis.direction, (list, tuple)):
                        anat_mat[:, i] = axis.direction
                    elif isinstance(axis.direction, (int, float)):
                        # Single value direction - expand to 3D vector
                        dir_vec = [0, 0, 0]
                        dir_vec[i] = axis.direction
                        anat_mat[:, i] = dir_vec
                    else:
                        anat_mat[:, i] = [0, 0, 0]
                        anat_mat[i, i] = 1
                
                # Compute similarity score (sum of absolute dot products)
                score = 0
                for i in range(3):
                    for j in range(3):
                        score += abs(np.dot(mat33[:, i], anat_mat[:, j]))
                
                if score > best_score:
                    best_score = score
                    best_orient = orient
            except:
                # Skip invalid orientations
                continue
        
        return find_anatomy_3d(best_orient)
    
    @property
    def ndim(self) -> int:
        """Number of dimensions.
        
        R Equivalent
        ------------
        neuroim2::ndim
        """
        return len(self.dim)
    
    @property
    def nvoxels(self) -> int:
        """Total number of voxels.
        
        R Equivalent
        ------------
        neuroim2::length (for NeuroSpace)
        """
        return int(np.prod(self.dim))
    
    def bounds(self) -> np.ndarray:
        """Get spatial bounds of the image space.
        
        Returns
        -------
        ndarray
            2 x ndim array with min bounds in first row, max in second row
            
        R Equivalent
        ------------
        neuroim2::bounds
        """
        # Grid corners in voxel space
        min_grid = np.zeros(self.ndim)
        max_grid = self.dim - 1
        
        # Convert to world coordinates
        min_coord = self.grid_to_coord(min_grid.reshape(1, -1))[0]
        max_coord = self.grid_to_coord(max_grid.reshape(1, -1))[0]
        
        # Stack as 2 x ndim array
        return np.vstack([min_coord, max_coord])
    
    def centroid(self) -> np.ndarray:
        """Get the spatial centroid of the image space.
        
        Returns
        -------
        ndarray
            Centroid coordinates in world space
            
        R Equivalent
        ------------
        neuroim2::centroid
        """
        center_grid = (self.dim - 1) / 2.0
        return self.grid_to_coord(center_grid.reshape(1, -1))[0]
    
    # Coordinate transformation methods
    def coord_to_grid(self, coords: np.ndarray) -> np.ndarray:
        """Convert real-world coordinates to grid indices.
        
        Parameters
        ----------
        coords : ndarray
            Real-world coordinates, shape (n, ndim) or (ndim,)
            
        Returns
        -------
        ndarray
            Grid indices (0-based), same shape as input
            
        Notes
        -----
        Unlike R which uses 1-based indexing, Python uses 0-based indexing.
        
        R Equivalent
        ------------
        neuroim2::coord_to_grid
        """
        coords = np.atleast_2d(coords)
        if coords.shape[1] != self.ndim:
            raise ValueError(f"Coordinates must have {self.ndim} dimensions, got {coords.shape[1]}")
        
        if self.ndim <= 3:
            # Apply inverse transformation
            homogeneous = np.column_stack([coords, np.ones(len(coords))])
            transformed = homogeneous @ self.inverse.T
            grid_coords = transformed[:, :self.ndim]
        else:
            # For 4D+ spaces, use simple scaling
            grid_coords = (coords - self.origin) / self.spacing

        # Return continuous grid coordinates (not rounded)
        # This allows accurate round-trip transformations
        return grid_coords
    
    def grid_to_coord(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid indices to real-world coordinates.

        Parameters
        ----------
        grid : ndarray
            Grid indices (0-based), shape (n, ndim) or (ndim,)

        Returns
        -------
        ndarray
            Real-world coordinates, same shape as input

        R Equivalent
        ------------
        neuroim2::grid_to_coord
        """
        grid = np.atleast_2d(grid)
        if grid.shape[1] != self.ndim:
            raise ValueError(f"Grid indices must have {self.ndim} dimensions, got {grid.shape[1]}")

        if self.ndim <= 3:
            # Apply transformation
            homogeneous = np.column_stack([grid, np.ones(len(grid))])
            transformed = homogeneous @ self.trans.T
            return transformed[:, :self.ndim]
        else:
            # For 4D+ spaces, use simple scaling
            return grid * self.spacing + self.origin

    def grid_to_world(self, coords):
        """Alias for grid_to_coord for API compatibility.

        Preserves input dimensionality: 1D input returns 1D output.
        """
        was_1d = np.ndim(coords) == 1
        result = self.grid_to_coord(np.atleast_2d(coords))
        return result[0] if was_1d else result

    def world_to_grid(self, coords):
        """Alias for coord_to_grid for API compatibility."""
        return self.coord_to_grid(coords)
    
    def coord_to_index(self, coords: np.ndarray) -> np.ndarray:
        """Convert real-world coordinates to 1D indices.
        
        Parameters
        ----------
        coords : ndarray
            Real-world coordinates, shape (n, ndim)
            
        Returns
        -------
        ndarray
            1D indices (0-based)
            
        R Equivalent
        ------------
        neuroim2::coord_to_index
        """
        grid = self.coord_to_grid(coords)
        grid_rounded = np.round(grid).astype(int)
        return self.grid_to_index(grid_rounded)
    
    def index_to_coord(self, idx: Union[int, np.ndarray]) -> np.ndarray:
        """Convert 1D indices to real-world coordinates.
        
        Parameters
        ----------
        idx : int or ndarray
            1D indices (0-based)
            
        Returns
        -------
        ndarray
            Real-world coordinates
            
        R Equivalent
        ------------
        neuroim2::index_to_coord
        """
        grid = self.index_to_grid(idx)
        return self.grid_to_coord(grid)
    
    def grid_to_index(self, grid: np.ndarray) -> np.ndarray:
        """Convert grid indices to 1D indices.
        
        Parameters
        ----------
        grid : ndarray
            Grid indices (0-based), shape (n, ndim)
            
        Returns
        -------
        ndarray
            1D indices (0-based)
            
        R Equivalent
        ------------
        neuroim2::grid_to_index
        """
        grid = np.atleast_2d(grid)
        if grid.shape[1] != self.ndim:
            raise ValueError(f"Grid indices must have {self.ndim} dimensions, got {grid.shape[1]}")
        
        # Use C-style (row-major) ordering to match R
        indices = np.zeros(len(grid), dtype=int)
        for i in range(len(grid)):
            indices[i] = np.ravel_multi_index(grid[i], self.dim, order='F')
        return indices
    
    def index_to_grid(self, idx: Union[int, np.ndarray]) -> np.ndarray:
        """Convert 1D indices to grid indices.
        
        Parameters
        ----------
        idx : int or ndarray
            1D indices (0-based)
            
        Returns
        -------
        ndarray
            Grid indices (0-based), shape (n, ndim)
            
        R Equivalent
        ------------
        neuroim2::index_to_grid
        """
        idx = np.atleast_1d(idx)
        
        # Use Fortran-style (column-major) ordering to match R
        grid_tuple = np.unravel_index(idx, self.dim, order='F')
        return np.column_stack(grid_tuple)
    
    def grid_to_grid(self, source_grid: np.ndarray, target_space: 'NeuroSpace') -> np.ndarray:
        """Transform grid coordinates from this space to another space.
        
        Parameters
        ----------
        source_grid : ndarray
            Grid indices in this space
        target_space : NeuroSpace
            Target space to transform to
            
        Returns
        -------
        ndarray
            Grid indices in target space
            
        R Equivalent
        ------------
        neuroim2::grid_to_grid
        """
        # Convert to world coordinates in source space
        coords = self.grid_to_coord(source_grid)
        # Convert to grid in target space
        return target_space.coord_to_grid(coords)
    
    # Dimension manipulation methods
    def dim_of(self, axis: Union[int, str, NamedAxis]) -> int:
        """Get the length of a given dimension.
        
        Parameters
        ----------
        axis : int, str, or NamedAxis
            Axis to query
            
        Returns
        -------
        int
            Length of the dimension
            
        R Equivalent
        ------------
        neuroim2::dim_of
        """
        if isinstance(axis, int):
            if axis < 0 or axis >= self.ndim:
                raise ValueError(f"Axis index {axis} out of range for {self.ndim}D space")
            return self.dim[axis]
        elif isinstance(axis, str):
            axis_idx = self.which_dim(axis)
            return self.dim[axis_idx]
        elif isinstance(axis, NamedAxis):
            axis_idx = self.which_dim(axis.axis)
            return self.dim[axis_idx]
        else:
            raise TypeError(f"axis must be int, str, or NamedAxis, got {type(axis)}")
    
    def which_dim(self, axis: Union[str, NamedAxis]) -> int:
        """Get the index of a dimension by axis name.
        
        Parameters
        ----------
        axis : str or NamedAxis
            Axis to find
            
        Returns
        -------
        int
            0-based index of the axis
            
        R Equivalent
        ------------
        neuroim2::which_dim
        """
        if isinstance(axis, NamedAxis):
            axis = axis.axis
            
        names = axis_names(self.axes)
        try:
            return names.index(axis)
        except ValueError:
            raise ValueError(f"Axis '{axis}' not found in space")
    
    def get_subspace(self, dims) -> 'NeuroSpace':
        """Return a subspace containing only the specified dimensions.

        Parameters
        ----------
        dims : iterable of int
            0-based indices of dimensions to keep.

        Returns
        -------
        NeuroSpace
            New space with only the requested dimensions.
        """
        dims = list(dims)
        if len(dims) == 0:
            raise ValueError("dims must include at least one dimension")
        for d in dims:
            if d < 0 or d >= self.ndim:
                raise ValueError(f"Dimension index {d} out of range for {self.ndim}D space")

        new_dim = self.dim[dims]
        new_spacing = self.spacing[dims]
        new_origin = self.origin[dims]
        ndim_new = len(dims)

        affine_col = self.trans.shape[0] - 1
        full_trans = np.eye(self.ndim + 1)
        base_dim = min(self.trans.shape[0], self.ndim + 1)
        full_trans[:base_dim, :base_dim] = self.trans[:base_dim, :base_dim]
        if self.trans.shape[0] > self.ndim + 1:
            full_trans[:, -1] = self.trans[: self.ndim + 1, -1]

        for i in range(3, self.ndim + 1):
            if i < len(self.spacing):
                full_trans[i, i] = self.spacing[i]

        keep = list(dims) + [affine_col]
        sub_trans = full_trans[np.ix_(keep, keep)]

        if ndim_new <= 3:
            new_trans = np.eye(4)
            new_trans[:ndim_new, :ndim_new] = sub_trans[:ndim_new, :ndim_new]
            new_trans[:ndim_new, 3] = sub_trans[:ndim_new, -1]
        else:
            new_trans = sub_trans

        return NeuroSpace(new_dim, new_spacing, new_origin, trans=new_trans)

    def drop_dim(self, dimnum: int) -> 'NeuroSpace':
        """Remove a dimension from the space.
        
        Parameters
        ----------
        dimnum : int
            0-based index of dimension to drop
            
        Returns
        -------
        NeuroSpace
            New space with dimension removed
            
        R Equivalent
        ------------
        neuroim2::drop_dim
        """
        if self.ndim < 2:
            raise ValueError("Cannot drop dimension from space with less than 2 dimensions")
        
        if dimnum < 0 or dimnum >= self.ndim:
            raise ValueError(f"dimnum {dimnum} out of range for {self.ndim}D space")
        
        # Create new arrays with dimension removed
        new_dim = np.delete(self.dim, dimnum)
        new_origin = np.delete(self.origin, dimnum)
        new_spacing = np.delete(self.spacing, dimnum)
        
        new_axes = axis_set(len(new_dim), [ax for i, ax in enumerate(list(self.axes)) if i != dimnum])

        affine_col = self.trans.shape[0] - 1
        full_trans = np.eye(self.ndim + 1)
        base_dim = min(self.trans.shape[0], self.ndim + 1)
        full_trans[:base_dim, :base_dim] = self.trans[:base_dim, :base_dim]
        if self.trans.shape[0] > self.ndim + 1:
            full_trans[:, -1] = self.trans[: self.ndim + 1, -1]

        for i in range(3, self.ndim + 1):
            if i < len(self.spacing):
                full_trans[i, i] = self.spacing[i]

        keep = [i for i in range(self.ndim) if i != dimnum] + [affine_col]
        sub_trans = full_trans[np.ix_(keep, keep)]

        if len(new_dim) <= 3:
            new_trans = np.eye(4)
            new_trans[:len(new_dim), :len(new_dim)] = sub_trans[:len(new_dim), :len(new_dim)]
            new_trans[:len(new_dim), 3] = sub_trans[:len(new_dim), -1]
        else:
            new_trans = sub_trans

        return NeuroSpace(new_dim, new_spacing, new_origin, new_axes, new_trans)
    
    def add_dim(self, n: int = 1, size: int = 1) -> 'NeuroSpace':
        """Add a dimension to the space.
        
        Parameters
        ----------
        n : int, optional
            Number of dimensions to add (default 1)
        size : int, optional
            Size of the new dimension(s) (default 1)
            
        Returns
        -------
        NeuroSpace
            New space with added dimension(s)
            
        R Equivalent
        ------------
        neuroim2::add_dim
        """
        if n <= 0:
            raise ValueError("n must be positive")
        
        # Extend arrays
        new_dim = np.append(self.dim, [size] * n)
        new_origin = np.append(self.origin, [0.0] * n)
        new_spacing = np.append(self.spacing, [1.0] * n)
        
        # Preserve existing axes and append labeled placeholder axes for each added dimension.
        start = len(self.axes) + 1
        extra_axes = [NamedAxis(f"v{start + i}", 1) for i in range(n)]
        new_axes = axis_set(len(new_dim), list(self.axes) + extra_axes)

        # Preserve affine transform behavior:
        # - 3D and below use fixed 4x4 transforms.
        # - 4D+ use (ndim+1)x(ndim+1) transforms, extending with identity rows/cols.
        source_dim = self.trans.shape[0] - 1
        if new_dim.size <= 3:
            new_trans = np.eye(4)
            keep = min(source_dim, new_dim.size)
            new_trans[:keep, :keep] = self.trans[:keep, :keep]
            new_trans[:keep, 3] = self.trans[:keep, -1]
            for i in range(keep, new_dim.size):
                new_trans[i, i] = new_spacing[i]
        else:
            new_trans = np.eye(len(new_dim) + 1)
            keep = min(source_dim, len(new_dim))
            new_trans[:keep, :keep] = self.trans[:keep, :keep]
            new_trans[:keep, -1] = self.trans[:keep, -1]
            for i in range(source_dim, len(new_dim)):
                new_trans[i, i] = new_spacing[i]
        
        return NeuroSpace(new_dim, new_spacing, new_origin, new_axes, new_trans)
    
    def __repr__(self):
        """String representation matching R's show method."""
        axis_str = ", ".join(axis_names(self.axes))
        # Convert numpy types to Python types for cleaner display
        dim_tuple = tuple(int(d) for d in self.dim)
        origin_tuple = tuple(float(o) for o in self.origin)
        spacing_tuple = tuple(float(s) for s in self.spacing)
        return (f"NeuroSpace(\n"
                f"  dim     : {dim_tuple}\n"
                f"  origin  : {origin_tuple}\n" 
                f"  spacing : {spacing_tuple}\n"
                f"  axes    : {axis_str}\n"
                f"  nvoxels : {self.nvoxels}\n"
                f")")
    
    def __eq__(self, other):
        """Check equality with another NeuroSpace."""
        if not isinstance(other, NeuroSpace):
            return False
        
        return (np.array_equal(self.dim, other.dim) and
                np.array_equal(self.origin, other.origin) and
                np.array_equal(self.spacing, other.spacing) and
                np.array_equal(self.trans, other.trans))


# Factory function to match R's constructor style
def neurospace(dim: Union[Tuple[int, ...], List[int]], 
               spacing: Optional[Union[Tuple[float, ...], List[float]]] = None,
               origin: Optional[Union[Tuple[float, ...], List[float]]] = None,
               axes: Optional[Union[AxisSet1D, AxisSet2D, AxisSet3D, AxisSet4D, AxisSet5D]] = None,
               trans: Optional[np.ndarray] = None) -> NeuroSpace:
    """Create a NeuroSpace object.
    
    This is a factory function that matches R's NeuroSpace() constructor style.
    
    R Equivalent
    ------------
    neuroim2::NeuroSpace
    """
    return NeuroSpace(dim, spacing, origin, axes, trans)
