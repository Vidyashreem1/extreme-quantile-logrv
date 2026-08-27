# Log-scale LRV simulation.
# Generate only Y = log X. Both RV and LRV estimators are reported only on
# the log-quantile scale for X: log q_RV and log q_LRV.

rm(list = ls())
set.seed(123)

script_path <- tryCatch(
  normalizePath(sys.frame(1)$ofile, winslash = "/"),
  error = function(e) NA_character_
)
if (is.na(script_path) || !nzchar(script_path)) {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg)) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1L]), winslash = "/")
  } else {
    script_path <- normalizePath("simulation_run_logscale.R", winslash = "/")
  }
}
setwd(dirname(script_path))


source("simulation_project_config.R")
source("simulation_estimators_logscale.R")

dir.create(SIM_CONFIG$output_dir, recursive = TRUE, showWarnings = FALSE)

alpha <- 1 / (SIM_CONFIG$alpha_denominator * SIM_CONFIG$n)
k_grid <- seq(
  max(2L, floor(SIM_CONFIG$k_min_fraction * SIM_CONFIG$n)),
  min(SIM_CONFIG$n - 2L, floor(SIM_CONFIG$k_max_fraction * SIM_CONFIG$n)),
  by = SIM_CONFIG$k_step
)

suppressPackageStartupMessages(library(reticulate))
Sys.setenv(
  OMP_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker),
  MKL_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker),
  OPENBLAS_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker),
  NUMEXPR_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker),
  TORCH_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker)
)
if (nzchar(SIM_CONFIG$python_path) && file.exists(SIM_CONFIG$python_path)) {
  use_python(SIM_CONFIG$python_path, required = TRUE)
}
source_python(file.path(getwd(), "nn_estimators_logscale.py"))
stopifnot(py_module_available("torch"))

run_one_sample <- function(y, logqtrue, model_name, spec, replication) {
  rows <- list()
  methods <- c("RV_Weissman", "RV_Corrected", "LRV_Weissman", "LRV_Corrected")

  for (method in methods) {
    fit <- select_logq_estimator(
      y = y,
      alpha = alpha,
      k_grid = k_grid,
      method = method,
      rho_hat = spec$rho
    )
    err <- compute_logscale_errors(fit$logqhat, logqtrue)
    rows[[length(rows) + 1L]] <- data.frame(
      replication = replication,
      model = model_name,
      base_dist = spec$dist,
      gamma = spec$gamma,
      rho = spec$rho,
      method = method,
      estimator_family = ifelse(grepl("^RV_", method), "RV_on_X_logscale", "LRV_on_logX"),
      nn_backend = "not_applicable",
      selected_k = fit$selected_k,
      selected_J = 0L,
      selected_batch_size = 0L,
      selected_loss = "not_applicable",
      logqtrue = logqtrue,
      logqhat = fit$logqhat,
      logq_rv = ifelse(grepl("^RV_", method), fit$logqhat, NA_real_),
      logq_lrv = ifelse(grepl("^LRV_", method), fit$logqhat, NA_real_),
      MedSElog = err[["MedSElog"]],
      MedSEloglog = err[["MedSEloglog"]],
      stringsAsFactors = FALSE
    )
  }

  nn_rv <- nn_rv_logq_estimator(
    y,
    alpha,
    as.integer(k_grid),
    as.integer(SIM_CONFIG$J_grid),
    as.integer(SIM_CONFIG$nn_batch_sizes),
    as.character(SIM_CONFIG$nn_losses),
    epochs = as.integer(SIM_CONFIG$nn_epochs),
    lr = SIM_CONFIG$nn_lr,
    seed = as.integer(replication)
  )
  nn_lrv <- nn_lrv_logq_estimator(
    y,
    alpha,
    as.integer(k_grid),
    as.integer(SIM_CONFIG$J_grid),
    as.integer(SIM_CONFIG$nn_batch_sizes),
    as.character(SIM_CONFIG$nn_losses),
    epochs = as.integer(SIM_CONFIG$nn_epochs),
    lr = SIM_CONFIG$nn_lr,
    seed = as.integer(replication)
  )

  for (method in c("NN_RV_logscale", "NN_LRV_logscale")) {
    fit <- if (method == "NN_RV_logscale") nn_rv else nn_lrv
    logqhat <- as.numeric(fit$logqhat)
    err <- compute_logscale_errors(logqhat, logqtrue)
    rows[[length(rows) + 1L]] <- data.frame(
      replication = replication,
      model = model_name,
      base_dist = spec$dist,
      gamma = spec$gamma,
      rho = spec$rho,
      method = method,
      estimator_family = ifelse(method == "NN_RV_logscale", "RV_on_X_logscale", "LRV_on_logX"),
      nn_backend = "torch",
      selected_k = as.integer(fit$selected_k),
      selected_J = as.integer(fit$selected_J),
      selected_batch_size = as.integer(fit$selected_batch_size),
      selected_loss = as.character(fit$selected_loss),
      logqtrue = logqtrue,
      logqhat = logqhat,
      logq_rv = ifelse(method == "NN_RV_logscale", logqhat, NA_real_),
      logq_lrv = ifelse(method == "NN_LRV_logscale", logqhat, NA_real_),
      MedSElog = err[["MedSElog"]],
      MedSEloglog = err[["MedSEloglog"]],
      stringsAsFactors = FALSE
    )
  }

  do.call(rbind, rows)
}

summarise_results <- function(raw_results) {
  id_cols <- c("model", "base_dist", "gamma", "rho", "method", "estimator_family", "nn_backend")
  groups <- split(seq_len(nrow(raw_results)), raw_results[id_cols], drop = TRUE)
  rows <- lapply(groups, function(idx) {
    block <- raw_results[idx, , drop = FALSE]
    data.frame(
      model = block$model[1L],
      base_dist = block$base_dist[1L],
      gamma = block$gamma[1L],
      rho = block$rho[1L],
      method = block$method[1L],
      estimator_family = block$estimator_family[1L],
      nn_backend = block$nn_backend[1L],
      MedSElog = safe_median(block$MedSElog),
      MedSEloglog = safe_median(block$MedSEloglog),
      selected_k = safe_median(block$selected_k),
      selected_J = safe_median(block$selected_J),
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  out[order(out$model, out$method), ]
}

wide_metric <- function(summary_data, metric) {
  methods <- unique(summary_data$method)
  id_cols <- c("model", "base_dist", "gamma", "rho")
  out <- unique(summary_data[, id_cols])
  for (method in methods) {
    tmp <- summary_data[summary_data$method == method, c(id_cols, metric)]
    names(tmp)[ncol(tmp)] <- method
    out <- merge(out, tmp, by = id_cols, all.x = TRUE)
  }
  values <- as.matrix(out[, methods, drop = FALSE])
  preferred_index <- apply(values, 1, function(row) {
    if (all(is.na(row))) return(NA_integer_)
    which.min(ifelse(is.na(row), Inf, row))
  })
  out$preferred <- ifelse(is.na(preferred_index), NA_character_, methods[preferred_index])
  out[order(out$model), ]
}

specs <- make_logscale_specs()
message("Running log-scale LRV simulations only.")
message("n = ", SIM_CONFIG$n, ", R = ", SIM_CONFIG$R, ", alpha = ", signif(alpha, 4))
message("k grid: ", min(k_grid), " to ", max(k_grid), "; J grid: ", paste(SIM_CONFIG$J_grid, collapse = ", "))
message("Parallel workers: ", SIM_CONFIG$parallel_workers,
        "; Python threads per worker: ", SIM_CONFIG$python_threads_per_worker)

worker_count <- max(1L, as.integer(SIM_CONFIG$parallel_workers))
if (worker_count > 1L) {
  suppressPackageStartupMessages(library(parallel))
  sim_base_dir <- getwd()
  cl <- parallel::makeCluster(worker_count, type = "PSOCK")
  on.exit(parallel::stopCluster(cl), add = TRUE)

  parallel::clusterExport(
    cl,
    varlist = c("SIM_CONFIG", "alpha", "k_grid", "run_one_sample", "sim_base_dir"),
    envir = .GlobalEnv
  )
  parallel::clusterEvalQ(cl, {
    setwd(sim_base_dir)
    source("simulation_estimators_logscale.R")
    Sys.setenv(
      OMP_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker),
      MKL_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker),
      OPENBLAS_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker),
      NUMEXPR_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker),
      TORCH_NUM_THREADS = as.character(SIM_CONFIG$python_threads_per_worker)
    )
    suppressPackageStartupMessages(library(reticulate))
    if (nzchar(SIM_CONFIG$python_path) && file.exists(SIM_CONFIG$python_path)) {
      use_python(SIM_CONFIG$python_path, required = TRUE)
    }
    source_python(file.path(getwd(), "nn_estimators_logscale.py"))
    stopifnot(py_module_available("torch"))
    NULL
  })
}

all_results <- list()
for (spec_id in seq_along(specs)) {
  spec <- specs[[spec_id]]
  model_name <- paste0("LRV_", spec$name)
  logqtrue <- logq_lrv(1 - alpha, spec)
  message("[", spec_id, "/", length(specs), "] ", model_name)

  if (worker_count > 1L) {
    model_results <- parallel::parLapply(
      cl,
      seq_len(SIM_CONFIG$R),
      function(replication, spec, spec_id, model_name, logqtrue) {
        set.seed(as.integer(SIM_CONFIG$seed + spec_id * 100000L + replication))
        y <- r_log_lrv(SIM_CONFIG$n, spec)
        run_one_sample(
          y = y,
          logqtrue = logqtrue,
          model_name = model_name,
          spec = spec,
          replication = replication
        )
      },
      spec = spec,
      spec_id = spec_id,
      model_name = model_name,
      logqtrue = logqtrue
    )
  } else {
    model_results <- vector("list", SIM_CONFIG$R)
    for (replication in seq_len(SIM_CONFIG$R)) {
      set.seed(as.integer(SIM_CONFIG$seed + spec_id * 100000L + replication))
      y <- r_log_lrv(SIM_CONFIG$n, spec)
      model_results[[replication]] <- run_one_sample(
        y = y,
        logqtrue = logqtrue,
        model_name = model_name,
        spec = spec,
        replication = replication
      )
      if (replication %% 25L == 0L) message("  replication ", replication, "/", SIM_CONFIG$R)
    }
  }

  model_df <- do.call(rbind, model_results)
  write.csv(model_df, file.path(SIM_CONFIG$output_dir, paste0(model_name, "_raw_logscale.csv")), row.names = FALSE)
  all_results[[length(all_results) + 1L]] <- model_df
}

raw_results <- do.call(rbind, all_results)
summary_long <- summarise_results(raw_results)
wide_MedSElog <- wide_metric(summary_long, "MedSElog")
wide_MedSEloglog <- wide_metric(summary_long, "MedSEloglog")

if (isTRUE(SIM_CONFIG$save_raw)) {
  write.csv(raw_results, file.path(SIM_CONFIG$output_dir, "raw_replications_logscale.csv"), row.names = FALSE)
}
write.csv(summary_long, file.path(SIM_CONFIG$output_dir, "summary_long_logscale.csv"), row.names = FALSE)
write.csv(wide_MedSElog, file.path(SIM_CONFIG$output_dir, "wide_MedSElog_logscale.csv"), row.names = FALSE)
write.csv(wide_MedSEloglog, file.path(SIM_CONFIG$output_dir, "wide_MedSEloglog_logscale.csv"), row.names = FALSE)

run_metadata <- data.frame(
  n = SIM_CONFIG$n,
  R = SIM_CONFIG$R,
  alpha = alpha,
  generated_data = "Y = log X only",
  target = "log q_X = q_Y",
  metrics = "MedSElog, MedSEloglog",
  python_path = SIM_CONFIG$python_path,
  J_grid = paste(SIM_CONFIG$J_grid, collapse = ","),
  nn_batch_sizes = paste(SIM_CONFIG$nn_batch_sizes, collapse = ","),
  nn_losses = paste(SIM_CONFIG$nn_losses, collapse = ","),
  parallel_workers = SIM_CONFIG$parallel_workers,
  python_threads_per_worker = SIM_CONFIG$python_threads_per_worker,
  stringsAsFactors = FALSE
)
write.csv(run_metadata, file.path(SIM_CONFIG$output_dir, "run_metadata_logscale.csv"), row.names = FALSE)

message("Done. Outputs written to: ", normalizePath(SIM_CONFIG$output_dir, winslash = "/"))

##plots

library(dplyr)
library(ggplot2)

plot_df <- raw_results %>%
  mutate(method = recode(method,
                         "NN_LRV_logscale" = "NN-LRV",
                         "NN_RV_logscale"  = "NN-RV",
                         "LRV_Weissman"    = "LRV-W",
                         "RV_Weissman"     = "RV-W",
                         "LRV_Corrected"   = "LRV-CW",
                         "RV_Corrected"    = "RV-CW"
  )) %>%
  filter(base_dist == "Burr", gamma == 1, rho == -1)
#boxplot


ggplot(plot_df,
       aes(x = method,
           y = selected_k,
           fill = method)) +
  geom_boxplot(
    width = 0.7,
    outlier.size = 0.8
  ) +
  theme_bw(base_size = 14) +
  labs(
    x = NULL,
    y = expression(hat(k))
  ) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank()
  )

###
tab <- table(raw_results$method, raw_results$selected_J)
round(prop.table(tab, 1) * 100,1)
