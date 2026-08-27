rm(list = ls())

# Covariates: longitude, latitude, mean altitude. Neighbours: Mahalanobis.
# Testing: the local maximum of each target grid is held out.


suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(processx))

script_path <- tryCatch(
  normalizePath(sys.frame(1)$ofile, winslash = "/"),
  error = function(e) NA_character_
)
if (is.na(script_path) || !nzchar(script_path)) {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg)) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1L]), winslash = "/")
  } else {
    script_path <- normalizePath("07.Conditional_CENN_LDNN_LogRainfall.R", winslash = "/")
  }
}

SCRIPT_DIR <- dirname(script_path)
BASE_DIR <- normalizePath(
  Sys.getenv("CENN_LDNN_BASE_DIR", unset = SCRIPT_DIR),
  winslash = "/",
  mustWork = FALSE
)
COORD_FILE <- Sys.getenv(
  "CENN_LDNN_COORD_FILE",
  unset = file.path(BASE_DIR, "data", "western_ghats_coordinates.csv")
)
RAIN_DIR <- Sys.getenv(
  "CENN_LDNN_RAIN_DIR",
  unset = file.path(BASE_DIR, "data", "Western ghats data")
)
OUT_DIR <- Sys.getenv(
  "CENN_LDNN_OUT_DIR",
  unset = file.path(BASE_DIR, "outputs", "cenn_ldnn_lograinfall")
)
PYTHON_PATH <- Sys.getenv("CENN_LDNN_PYTHON", unset = unname(Sys.which("python")))
PY_ENGINE <- file.path(SCRIPT_DIR, "cenn_ldnn_lograinfall_engine.py")

SAMPLE_PER_REGION <- as.integer(Sys.getenv("CENN_LDNN_SAMPLE_PER_REGION", "30"))
REGION_ALT_BREAKS <- as.numeric(strsplit(Sys.getenv("CENN_LDNN_REGION_ALT_BREAKS", "-Inf,100,600,Inf"), ",", fixed = TRUE)[[1L]])
REGION_LABELS <- c("coastal", "mid_range", "hilly")
M_NEIGHBORS <- as.integer(strsplit(Sys.getenv("CENN_LDNN_M_NEIGHBORS", "25"), ",", fixed = TRUE)[[1L]])
K_GRID <- as.integer(strsplit(Sys.getenv("CENN_LDNN_K_GRID", "25,50,75,100,150,200"), ",", fixed = TRUE)[[1L]])
TAU_GRID <- as.integer(strsplit(Sys.getenv("CENN_LDNN_TAU_GRID", "300,500,750,1000,1500,2000"), ",", fixed = TRUE)[[1L]])
ANCHOR_GRID <- as.integer(strsplit(Sys.getenv("CENN_LDNN_ANCHOR_GRID", "1,2,3,5,10"), ",", fixed = TRUE)[[1L]])
J_GRID <- as.integer(strsplit(Sys.getenv("CENN_LDNN_J_GRID", "2,3"), ",", fixed = TRUE)[[1L]])
BATCH_SIZES <- as.integer(strsplit(Sys.getenv("CENN_LDNN_BATCH_SIZES", "512"), ",", fixed = TRUE)[[1L]])
LOSSES <- strsplit(Sys.getenv("CENN_LDNN_LOSSES", "l1"), ",", fixed = TRUE)[[1L]]
TAIL_SCALES <- toupper(strsplit(Sys.getenv("CENN_LDNN_TAIL_SCALES", "LRV,RV"), ",", fixed = TRUE)[[1L]])
NN_EPOCHS <- as.integer(Sys.getenv("CENN_LDNN_EPOCHS", "50"))
NN_LR <- as.numeric(Sys.getenv("CENN_LDNN_LR", "0.005"))
SEED <- as.integer(Sys.getenv("CENN_LDNN_SEED", "20260623"))
PARALLEL_GRID_RUN <- as.logical(Sys.getenv("CENN_LDNN_PARALLEL", "TRUE"))
AVAILABLE_CORES <- max(1L, parallel::detectCores(logical = FALSE))
DEFAULT_WORKERS <- max(1L, min(4L, AVAILABLE_CORES - 2L))
GRID_WORKERS <- as.integer(Sys.getenv("CENN_LDNN_WORKERS", as.character(DEFAULT_WORKERS)))
PYTHON_THREADS_PER_WORKER <- as.integer(Sys.getenv("CENN_LDNN_PYTHON_THREADS", "2"))
PROCESS_ECHO <- as.logical(Sys.getenv("CENN_LDNN_PROCESS_ECHO", "FALSE"))

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "inputs"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "candidates"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(OUT_DIR, "selected"), recursive = TRUE, showWarnings = FALSE)

if (!nzchar(PYTHON_PATH) || !file.exists(PYTHON_PATH)) {
  stop("Python executable not found. Set CENN_LDNN_PYTHON to an environment containing NumPy and PyTorch.")
}
if (!file.exists(PY_ENGINE)) stop("Python engine not found: ", PY_ENGINE)

run_python_engine <- function(args, label) {
  result <- processx::run(
    command = PYTHON_PATH,
    args = args,
    wd = SCRIPT_DIR,
    echo = PROCESS_ECHO,
    env = c(
      OMP_NUM_THREADS = as.character(PYTHON_THREADS_PER_WORKER),
      MKL_NUM_THREADS = as.character(PYTHON_THREADS_PER_WORKER),
      OPENBLAS_NUM_THREADS = as.character(PYTHON_THREADS_PER_WORKER),
      NUMEXPR_NUM_THREADS = as.character(PYTHON_THREADS_PER_WORKER),
      TORCH_NUM_THREADS = as.character(PYTHON_THREADS_PER_WORKER)
    ),
    error_on_status = FALSE
  )
  if (!identical(result$status, 0L)) {
    stop(
      "Python engine failed for ", label,
      "\nExit status: ", result$status,
      "\nstdout:\n", result$stdout,
      "\nstderr:\n", result$stderr
    )
  }
  invisible(result)
}

grid_file_name <- function(lat, lon) paste0("data_", lat, "_", lon, ".csv")
grid_id <- function(lat, lon) paste(lat, lon, sep = "_")

read_coordinates <- function(coord_file) {
  coords <- fread(coord_file)
  required <- c("lat", "lon", "alt_mean")
  missing_cols <- setdiff(required, names(coords))
  if (length(missing_cols)) stop("Coordinate file is missing: ", paste(missing_cols, collapse = ", "))
  coords[, (required) := lapply(.SD, as.numeric), .SDcols = required]
  coords[, grid_id := grid_id(lat, lon)]
  coords[]
}

add_standardized_covariates <- function(coords) {
  out <- copy(coords)
  covariates <- c("lon", "lat", "alt_mean")
  for (v in covariates) {
    mu <- mean(out[[v]], na.rm = TRUE)
    sig <- stats::sd(out[[v]], na.rm = TRUE)
    if (!is.finite(sig) || sig <= 0) sig <- 1
    out[, paste0(v, "_std") := (get(v) - mu) / sig]
  }
  setnames(out, c("lon_std", "lat_std"), c("lon_std_tmp", "lat_std_tmp"))
  setnames(out, c("lat_std_tmp", "lon_std_tmp"), c("lat_std", "lon_std"))
  out[]
}

assign_regions <- function(coords) {
  if (length(REGION_ALT_BREAKS) != 4L) {
    stop("CENN_LDNN_REGION_ALT_BREAKS must contain four comma-separated break points.")
  }
  out <- copy(coords)
  out[, region := cut(
    alt_mean,
    breaks = REGION_ALT_BREAKS,
    labels = REGION_LABELS,
    include.lowest = TRUE,
    right = FALSE
  )]
  if (anyNA(out$region)) stop("Some grids could not be assigned to altitude regions.")
  out[]
}

sample_target_grids <- function(coords, sample_per_region, seed) {
  set.seed(seed)
  coords[
    ,
    .SD[sample.int(.N, min(sample_per_region, .N))],
    by = region
  ][order(region, lat, lon)]
}

mahalanobis_neighbors <- function(coords, target_row, m_neighbors) {
  covariates <- c("lon", "lat", "alt_mean")
  cov_matrix <- stats::cov(coords[, ..covariates], use = "complete.obs")
  inv_cov <- tryCatch(solve(cov_matrix), error = function(e) MASS::ginv(cov_matrix))
  target_vec <- as.numeric(target_row[, ..covariates])
  cov_values <- as.matrix(coords[, ..covariates])
  centered <- sweep(cov_values, 2L, target_vec, "-")
  out <- copy(coords)
  out[, mahalanobis_distance := sqrt(rowSums((centered %*% inv_cov) * centered))]
  out[order(mahalanobis_distance)][seq_len(min(m_neighbors, .N))]
}

read_one_grid <- function(file) {
  name <- tools::file_path_sans_ext(basename(file))
  parts <- strsplit(name, "_", fixed = TRUE)[[1L]]
  if (length(parts) != 3L) stop("Unexpected rainfall filename: ", basename(file))
  dt <- fread(
    file,
    header = FALSE,
    col.names = c("Year", "Month", "Day", "Precipitation", "MaxTemp", "MinTemp", "MeanTemp")
  )
  dt[, `:=`(
    lat = as.numeric(parts[2L]),
    lon = as.numeric(parts[3L]),
    Date = as.Date(sprintf("%04d-%02d-%02d", Year, Month, Day))
  )]
  dt[]
}

read_neighbor_rainfall <- function(neighbors) {
  neighbors <- copy(neighbors)
  neighbors[, file_path := file.path(RAIN_DIR, grid_file_name(lat, lon))]
  missing_files <- neighbors[!file.exists(file_path), file_path]
  if (length(missing_files)) stop("Missing rainfall files:\n", paste(missing_files, collapse = "\n"))
  rain <- rbindlist(lapply(neighbors$file_path, read_one_grid), use.names = TRUE)
  merge(
    rain,
    neighbors[, .(lat, lon, grid_id, region, alt_mean, lat_std, lon_std, alt_mean_std, mahalanobis_distance)],
    by = c("lat", "lon"),
    all.x = TRUE,
    sort = FALSE
  )
}

hold_out_local_maxima <- function(dt) {
  positive <- dt[is.finite(Precipitation) & Precipitation > 0]
  positive[, row_id := seq_len(.N)]
  holdout_ids <- positive[
    ,
    .SD[which.max(Precipitation)][1L, .(row_id, heldout_date = Date, heldout_rainfall = Precipitation)],
    by = grid_id
  ]
  train <- positive[!row_id %in% holdout_ids$row_id]
  train[, Y := log(Precipitation)]
  list(train = train, holdouts = holdout_ids)
}

prepare_target_neighborhood <- function(coords, target_row, m_neighbors) {
  target_grid <- target_row$grid_id
  slug <- safe_slug(target_grid, m_neighbors)
  neighbors <- mahalanobis_neighbors(coords, target_row, m_neighbors)
  fwrite(neighbors, file.path(OUT_DIR, paste0(slug, "_neighbors.csv")))

  rain <- read_neighbor_rainfall(neighbors)
  positive <- rain[is.finite(Precipitation) & Precipitation > 0]
  if (nrow(positive) <= max(K_GRID, na.rm = TRUE) + 2L) {
    warning("Skipping ", target_grid, ": pooled neighbourhood sample is too small for the requested k grid.")
    return(NULL)
  }

  positive[, row_id := seq_len(.N)]
  holdout <- positive[which.max(Precipitation)][1L]
  train <- positive[row_id != holdout$row_id]
  target_cov <- target_row[, .(lat_std, lon_std, alt_mean_std)]

  input <- train[
    ,
    .(
      Y = log(Precipitation),
      lat_std = target_cov$lat_std,
      lon_std = target_cov$lon_std,
      alt_mean_std = target_cov$alt_mean_std,
      grid_id = target_grid
    )
  ]

  metadata <- data.table(
    target_grid_id = target_grid,
    target_lat = target_row$lat,
    target_lon = target_row$lon,
    target_region = as.character(target_row$region),
    target_alt_mean = target_row$alt_mean,
    neighbor_count = m_neighbors,
    heldout_date = holdout$Date,
    heldout_rainfall = holdout$Precipitation,
    target_train_n = nrow(train)
  )
  list(input = input, metadata = metadata)
}

safe_slug <- function(target_grid_id, m_neighbors) {
  paste0("target_", gsub("[.]", "p", target_grid_id), "_m", m_neighbors)
}

run_one_target <- function(coords, target_row, m_neighbors,
                           global_input = NULL,
                           global_metadata = NULL,
                           global_input_csv = NULL) {
  target_grid <- target_row$grid_id
  slug <- safe_slug(target_grid, m_neighbors)
  message("Running ", slug)

  if (is.null(global_metadata) || is.null(global_input_csv)) {
    prepared <- prepare_target_neighborhood(coords, target_row, m_neighbors)
    if (is.null(prepared)) return(NULL)
    target_meta <- prepared$metadata
  } else {
    target_meta <- global_metadata[target_grid_id == target_grid]
    if (nrow(target_meta) != 1L) {
      warning("Skipping ", target_grid, ": target metadata not found in global neighbourhood panel.")
      return(NULL)
    }
  }

  alpha_test <- 1 / (target_meta$target_train_n + 1)
  target_cov <- target_row[, .(lat_std, lon_std, alt_mean_std)]
  valid_k_grid <- K_GRID[K_GRID < target_meta$target_train_n]
  if (!length(valid_k_grid)) {
    warning("Skipping ", target_grid, ": no valid k values for target training size.")
    return(NULL)
  }
  valid_tau_grid <- sort(unique(c(
    TAU_GRID,
    2L * valid_k_grid,
    floor(target_meta$target_train_n / 2),
    target_meta$target_train_n - 1L
  )))
  valid_tau_grid <- valid_tau_grid[valid_tau_grid > min(valid_k_grid, na.rm = TRUE) & valid_tau_grid < target_meta$target_train_n]
  if (!length(valid_tau_grid)) {
    warning("Skipping ", target_grid, ": no valid LDNN tau values for target training size.")
    return(NULL)
  }
  input_csv <- if (!is.null(global_input_csv)) {
    global_input_csv
  } else {
    file.path(OUT_DIR, "inputs", paste0(slug, "_input.csv"))
  }
  candidate_csv <- file.path(OUT_DIR, "candidates", paste0(slug, "_candidates.csv"))
  selected_csv <- file.path(OUT_DIR, "selected", paste0(slug, "_selected.csv"))
  if (is.null(global_input_csv)) fwrite(prepared$input, input_csv)

  args <- c(
    PY_ENGINE,
    "--input-csv", input_csv,
    "--candidate-output-csv", candidate_csv,
    "--selected-output-csv", selected_csv,
    "--target-grid-id", target_grid,
    paste0("--target-covariate=", paste(as.numeric(unlist(target_cov[1L])), collapse = ",")),
    "--heldout-log-max", as.character(log(target_meta$heldout_rainfall)),
    "--alpha-test", as.character(alpha_test),
    "--k-grid", paste(valid_k_grid, collapse = ","),
    "--tau-grid", paste(valid_tau_grid, collapse = ","),
    "--anchor-grid", paste(ANCHOR_GRID, collapse = ","),
    "--j-grid", paste(J_GRID, collapse = ","),
    "--batch-sizes", paste(BATCH_SIZES, collapse = ","),
    "--losses", paste(LOSSES, collapse = ","),
    "--tail-scales", paste(TAIL_SCALES, collapse = ","),
    "--epochs", as.character(NN_EPOCHS),
    "--lr", as.character(NN_LR),
    "--seed", as.character(SEED)
  )
  run_python_engine(args, slug)

  selected <- fread(selected_csv)
  selected[, `:=`(
    target_grid_id = target_meta$target_grid_id,
    target_lat = target_meta$target_lat,
    target_lon = target_meta$target_lon,
    target_region = target_meta$target_region,
    target_alt_mean = target_meta$target_alt_mean,
    neighbor_count = target_meta$neighbor_count,
    heldout_date = target_meta$heldout_date,
    heldout_rainfall = target_meta$heldout_rainfall,
    target_train_n = target_meta$target_train_n
  )]
  candidate <- fread(candidate_csv)
  candidate[, `:=`(
    target_grid_id = target_grid,
    target_region = as.character(target_row$region),
    neighbor_count = m_neighbors,
    target_train_n = target_meta$target_train_n
  )]
  fwrite(selected, selected_csv)
  fwrite(candidate, candidate_csv)
  list(selected = selected, candidate = candidate)
}

run_targets_for_m <- function(coords, targets, m, global_metadata, global_input_csv) {
  worker_count <- min(GRID_WORKERS, nrow(targets))
  if (!isTRUE(PARALLEL_GRID_RUN) || worker_count <= 1L) {
    return(lapply(seq_len(nrow(targets)), function(i) {
      run_one_target(
        coords,
        targets[i],
        m,
        global_metadata = global_metadata,
        global_input_csv = global_input_csv
      )
    }))
  }

  message("Running ", nrow(targets), " target grids with ", worker_count, " parallel workers.")
  cl <- parallel::makeCluster(worker_count, type = "PSOCK")
  on.exit(parallel::stopCluster(cl), add = TRUE)
  parallel::clusterEvalQ(cl, {
    user_r_libs <- Sys.glob(file.path(Sys.getenv("USERPROFILE"), "Documents", "R", "win-library", "*"))
    if (length(user_r_libs)) .libPaths(c(user_r_libs, .libPaths()))
    suppressPackageStartupMessages(library(data.table))
    NULL
  })
  parallel::clusterExport(
    cl,
    varlist = ls(envir = .GlobalEnv),
    envir = .GlobalEnv
  )
  parallel::parLapply(cl, seq_len(nrow(targets)), function(i) {
    run_one_target(
      coords,
      targets[i],
      m,
      global_metadata = global_metadata,
      global_input_csv = global_input_csv
    )
  })
}

summarise_selected <- function(selected) {
  selected[
    ,
    .(
      grids = .N,
      mean_MSE_log = mean(MSE_log, na.rm = TRUE),
      median_MSE_log = median(MSE_log, na.rm = TRUE),
      mean_MSE_loglog = mean(MSE_loglog, na.rm = TRUE),
      median_MSE_loglog = median(MSE_loglog, na.rm = TRUE),
      median_anchor_MAD = median(anchor_mad, na.rm = TRUE),
      median_k_delta = median(k_delta, na.rm = TRUE),
      median_J = median(J, na.rm = TRUE)
    ),
    by = .(target_region, neighbor_count, model)
  ][order(target_region, neighbor_count, model)]
}

run_pipeline <- function() {
  coords <- add_standardized_covariates(assign_regions(read_coordinates(COORD_FILE)))
  targets <- sample_target_grids(coords, SAMPLE_PER_REGION, SEED)
  fwrite(targets, file.path(OUT_DIR, "sampled_target_grids.csv"))

  metadata <- data.table(
    created = as.character(Sys.time()),
    sample_per_region = SAMPLE_PER_REGION,
    region_alt_breaks = paste(REGION_ALT_BREAKS, collapse = ","),
    m_neighbors = paste(M_NEIGHBORS, collapse = ","),
    k_grid = paste(K_GRID, collapse = ","),
    tau_grid = paste(TAU_GRID, collapse = ","),
    anchor_grid = paste(ANCHOR_GRID, collapse = ","),
    j_grid = paste(J_GRID, collapse = ","),
    batch_sizes = paste(BATCH_SIZES, collapse = ","),
    losses = paste(LOSSES, collapse = ","),
    tail_scales = paste(TAIL_SCALES, collapse = ","),
    epochs = NN_EPOCHS,
    learning_rate = NN_LR,
    seed = SEED,
    parallel_grid_run = PARALLEL_GRID_RUN,
    grid_workers = GRID_WORKERS,
    python_threads_per_worker = PYTHON_THREADS_PER_WORKER,
    python_path = PYTHON_PATH
  )
  fwrite(metadata, file.path(OUT_DIR, "run_metadata.csv"))

  results <- list()
  counter <- 0L
  for (m in M_NEIGHBORS) {
    message("Preparing global neighbourhood panel for m = ", m)
    prepared <- lapply(seq_len(nrow(targets)), function(i) prepare_target_neighborhood(coords, targets[i], m))
    prepared <- Filter(Negate(is.null), prepared)
    if (!length(prepared)) next
    global_input <- rbindlist(lapply(prepared, `[[`, "input"), fill = TRUE)
    global_metadata <- rbindlist(lapply(prepared, `[[`, "metadata"), fill = TRUE)
    global_input_csv <- file.path(OUT_DIR, "inputs", paste0("global_m", m, "_input.csv"))
    fwrite(global_input, global_input_csv)
    fwrite(global_metadata, file.path(OUT_DIR, paste0("global_m", m, "_target_metadata.csv")))

    target_results <- run_targets_for_m(
      coords = coords,
      targets = targets,
      m = m,
      global_metadata = global_metadata,
      global_input_csv = global_input_csv
    )
    target_results <- Filter(Negate(is.null), target_results)
    if (length(target_results)) {
      results <- c(results, target_results)
    }
  }
  if (!length(results)) stop("No target grids completed.")

  all_selected <- rbindlist(lapply(results, `[[`, "selected"), fill = TRUE)
  all_candidates <- rbindlist(lapply(results, `[[`, "candidate"), fill = TRUE)
  summary <- summarise_selected(all_selected)

  fwrite(all_selected, file.path(OUT_DIR, "all_selected_models.csv"))
  fwrite(all_candidates, file.path(OUT_DIR, "all_candidate_models.csv"))
  fwrite(summary, file.path(OUT_DIR, "summary_by_region_neighbor_model.csv"))

  message("Done. Outputs written to: ", normalizePath(OUT_DIR, winslash = "/"))
  invisible(list(selected = all_selected, candidates = all_candidates, summary = summary))
}

if (!identical(Sys.getenv("RUN_CENN_LDNN_LOGRAINFALL", unset = "1"), "0")) {
  pipeline_results <- run_pipeline()
}


summarise_selected_overall <- function(selected) {
  selected[
    ,
    .(
      grids = .N,
      mean_MSE_log = mean(MSE_log, na.rm = TRUE),
      median_MSE_log = median(MSE_log, na.rm = TRUE),
      mean_MSE_loglog = mean(MSE_loglog, na.rm = TRUE),
      median_MSE_loglog = median(MSE_loglog, na.rm = TRUE),
      median_anchor_MAD = median(anchor_mad, na.rm = TRUE),
      median_k_delta = median(k_delta, na.rm = TRUE),
      median_J = median(J, na.rm = TRUE)
    ),
    by = .(model)
  ][order(model)]
}

selected_summary_file <- file.path(OUT_DIR, "all_selected_models.csv")
if (file.exists(selected_summary_file)) {
  all_selected <- fread(selected_summary_file)
  overall_summary <- summarise_selected_overall(all_selected)
  print(overall_summary)
  fwrite(overall_summary, file.path(OUT_DIR, "overall_summary.csv"))
}
