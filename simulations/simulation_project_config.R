# Simulation configuration for RV/LRV extreme quantile comparisons.

SIM_CONFIG <- list(
  n = 500L,
  R = 200L,
  alpha_denominator = 2,
  seed = 123L,
  output_dir = file.path(getwd(), "simulation_outputs_rv_lrv_comparison"),

  # Python environment where Torch is installed. Override with SIM_PYTHON.
  python_path = unname(Sys.which("python")),


  nn_epochs = 50L,
  nn_lr = 0.01,
  nn_batch_sizes = c(512L),
  nn_losses = c("l1"),
  J_grid = 2L:3L,
  parallel_workers = 4L,
  python_threads_per_worker = 1L,

 # intermediate sequence used for estimator selection.
  k_min_fraction = 0.03,
  k_max_fraction = 0.50,
  k_step = 5L,
  save_raw = TRUE

)

env_logical <- function(name, default) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) return(default)
  tolower(value) %in% c("1", "true", "yes", "y")
}

env_integer <- function(name, default) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) return(default)
  as.integer(value)
}

env_integer_vector <- function(name, default) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) return(default)
  as.integer(strsplit(value, ",", fixed = TRUE)[[1L]])
}

env_character_vector <- function(name, default) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) return(default)
  trimws(strsplit(value, ",", fixed = TRUE)[[1L]])
}

env_character <- function(name, default) {
  value <- Sys.getenv(name, unset = NA_character_)
  if (is.na(value) || !nzchar(value)) return(default)
  value
}

SIM_CONFIG$R <- env_integer("SIM_R", SIM_CONFIG$R)
SIM_CONFIG$n <- env_integer("SIM_N", SIM_CONFIG$n)
SIM_CONFIG$python_path <- env_character("SIM_PYTHON", SIM_CONFIG$python_path)
SIM_CONFIG$nn_epochs <- env_integer("SIM_NN_EPOCHS", SIM_CONFIG$nn_epochs)
SIM_CONFIG$J_grid <- env_integer_vector("SIM_J_GRID", SIM_CONFIG$J_grid)
SIM_CONFIG$nn_batch_sizes <- env_integer_vector("SIM_NN_BATCH_SIZES", SIM_CONFIG$nn_batch_sizes)
SIM_CONFIG$nn_losses <- env_character_vector("SIM_NN_LOSSES", SIM_CONFIG$nn_losses)
SIM_CONFIG$parallel_workers <- env_integer("SIM_PARALLEL_WORKERS", SIM_CONFIG$parallel_workers)
SIM_CONFIG$python_threads_per_worker <- env_integer("SIM_PYTHON_THREADS", SIM_CONFIG$python_threads_per_worker)


# For a quick smoke test before a full run
# SIM_CONFIG$R <- 5L
# SIM_CONFIG$include_nn <- TRUE
# SIM_CONFIG$nn_epochs <- 20L
# SIM_CONFIG$nn_batch_sizes <- 256L
# SIM_CONFIG$nn_losses <- "l1"
