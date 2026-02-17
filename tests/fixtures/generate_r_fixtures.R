# Script to generate R fixtures for Python equivalence testing
# Run this in R to create test data that Python tests will validate against

library(neuroim2)
set.seed(42)  # Ensure reproducibility

# Create output directory
dir.create("r_outputs", showWarnings = FALSE)

# Test 1: Basic NeuroVol operations
cat("Generating NeuroVol fixtures...\n")
space <- NeuroSpace(dim = c(10, 10, 10), 
                   spacing = c(2, 2, 2),
                   origin = c(0, 0, 0))

# Create test data with known values
test_data <- array(1:1000, dim = c(10, 10, 10))
vol <- NeuroVol(test_data, space)

# Save the volume
saveRDS(vol, "r_outputs/test_vol.rds")

# Test arithmetic operations
vol_add <- vol + 10
vol_mult <- vol * 2
vol_div <- vol / 2

saveRDS(vol_add@.Data, "r_outputs/vol_add_result.rds")
saveRDS(vol_mult@.Data, "r_outputs/vol_mult_result.rds") 
saveRDS(vol_div@.Data, "r_outputs/vol_div_result.rds")

# Test 2: NeuroVec time series extraction
cat("Generating NeuroVec fixtures...\n")
vec_space <- NeuroSpace(dim = c(10, 10, 10, 20))
vec_data <- array(rnorm(10*10*10*20), dim = c(10, 10, 10, 20))
vec <- NeuroVec(vec_data, vec_space)

saveRDS(vec, "r_outputs/test_vec.rds")

# Extract time series from specific voxels
ts1 <- series(vec, c(5, 5, 5))
ts2 <- series(vec, matrix(c(3,3,3, 7,7,7, 5,6,7), ncol=3, byrow=TRUE))

saveRDS(ts1, "r_outputs/vec_series_single.rds")
saveRDS(ts2, "r_outputs/vec_series_multi.rds")

# Test 3: SparseNeuroVec operations
cat("Generating SparseNeuroVec fixtures...\n")
mask_data <- array(runif(10*10*10) > 0.7, dim = c(10, 10, 10))
mask <- LogicalNeuroVol(mask_data, space)

sparse_data <- matrix(rnorm(sum(mask) * 20), nrow = sum(mask), ncol = 20)
sparse_vec <- SparseNeuroVec(data = sparse_data, 
                            mask = mask,
                            space = vec_space)

saveRDS(mask, "r_outputs/test_mask.rds")
saveRDS(sparse_vec, "r_outputs/test_sparse_vec.rds")

# Extract series from sparse vector
sparse_ts <- series(sparse_vec, c(5, 5, 5))
saveRDS(sparse_ts, "r_outputs/sparse_vec_series.rds")

# Test 4: ROI operations
cat("Generating ROI fixtures...\n")
roi_sphere <- spherical_roi(c(5, 5, 5), radius = 3, space = space)
roi_cube <- cubic_roi(c(5, 5, 5), surround = 2, space = space)

saveRDS(coords(roi_sphere), "r_outputs/roi_sphere_coords.rds")
saveRDS(coords(roi_cube), "r_outputs/roi_cube_coords.rds")

# Test 5: Connected components
cat("Generating connected components fixtures...\n")
stat_data <- array(rnorm(10*10*10, mean = 0, sd = 2), dim = c(10, 10, 10))
stat_vol <- NeuroVol(stat_data, space)

cc_result <- conn_comp(stat_vol, threshold = 1.5, connect = "26-connect")
saveRDS(cc_result$mask@.Data, "r_outputs/conn_comp_mask.rds")
saveRDS(cc_result$cluster_map@.Data, "r_outputs/conn_comp_clusters.rds")
saveRDS(cc_result$cluster_table, "r_outputs/conn_comp_table.rds")

# Test 6: Searchlight (simple example)
cat("Generating searchlight fixtures...\n")
# Create a simple function that returns mean value
search_fun <- function(x) {
  if (ncol(x) == 0) return(NA)
  mean(x, na.rm = TRUE)
}

# Small searchlight for testing
search_result <- searchlight(mask, radius = 2, 
                           method = search_fun,
                           combiner = "mean")

saveRDS(search_result@.Data, "r_outputs/searchlight_result.rds")

# Test 7: Statistical operations
cat("Generating statistical operation fixtures...\n")
# Partition
partitioned <- partition(vol, k = 3, mask = mask)
saveRDS(partitioned@.Data, "r_outputs/partition_result.rds")

# Split blocks
indices <- c(1, 100, 200, 300, 400, 500)
block_ids <- c(1, 1, 2, 2, 3, 3)
blocks <- split_blocks(vol, indices = indices, block_ids = block_ids)
saveRDS(length(blocks), "r_outputs/split_blocks_count.rds")
saveRDS(lapply(blocks, function(b) dim(b)), "r_outputs/split_blocks_dims.rds")

# Save metadata
cat("Saving metadata...\n")
metadata <- list(
  r_version = as.character(getRversion()),
  neuroim2_version = packageVersion("neuroim2"),
  seed = 42,
  creation_date = Sys.Date()
)
saveRDS(metadata, "r_outputs/metadata.rds")

cat("R fixtures generated successfully!\n")
cat("Files saved in: r_outputs/\n")