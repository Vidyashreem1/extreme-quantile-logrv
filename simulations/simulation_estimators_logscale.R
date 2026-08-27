# Estimators for the log-scale LRV simulation.
# We simulate only Y = log X. The target is log q_X(1-alpha) = q_Y(1-alpha).
# All estimators below return a log-quantile estimate

safe_median <- function(x) {
  x <- x[is.finite(x)]
  if (!length(x)) return(NA_real_)
  median(x)
}

safe_log <- function(x) {
  ifelse(is.finite(x) & x > 0, log(x), NA_real_)
}

hill_estimator <- function(z, k) {
  z <- sort(z[is.finite(z) & z > 0], decreasing = TRUE)
  n <- length(z)
  if (k < 2 || k >= n) return(NA_real_)
  threshold <- z[k + 1L]
  if (!is.finite(threshold) || threshold <= 0) return(NA_real_)
  mean(log(z[seq_len(k)] / threshold))
}

estimate_A <- function(z, k, rho_hat) {
  n <- length(z)
  if (2L * k >= n || !is.finite(rho_hat) || abs(rho_hat) < 1e-10) return(NA_real_)
  h_k <- hill_estimator(z, k)
  h_2k <- hill_estimator(z, 2L * k)
  denom <- 1 - 2^(-rho_hat)
  if (!is.finite(h_k) || !is.finite(h_2k) || abs(denom) < 1e-10) return(NA_real_)
  (1 - rho_hat) * (h_k - h_2k) / denom
}

# RV estimator on the original X scale, computed without forming X = exp(Y):
# log qhat_X = Y_{n-k,n} + gamma_H(X) log(k/(n alpha)),
# and gamma_H(X) = mean(Y_i - Y_{n-k,n}).
rv_log_weissman <- function(y, k, alpha) {
  y <- sort(y[is.finite(y)], decreasing = TRUE)
  n <- length(y)
  if (k < 2 || k >= n) return(NA_real_)
  anchor <- y[k + 1L]
  gamma_hat_x <- mean(y[seq_len(k)] - anchor)
  if (!is.finite(anchor) || !is.finite(gamma_hat_x)) return(NA_real_)
  anchor + gamma_hat_x * log(k / (n * alpha))
}

rv_log_corrected_weissman <- function(y, k, alpha, rho_hat) {
  y <- sort(y[is.finite(y)], decreasing = TRUE)
  n <- length(y)
  if (k < 2 || k >= n || 2L * k >= n) return(NA_real_)
  anchor <- y[k + 1L]
  gamma_hat_x <- mean(y[seq_len(k)] - anchor)
  gamma_hat_2k_x <- mean(y[seq_len(2L * k)] - y[2L * k + 1L])
  denom <- 1 - 2^(-rho_hat)
  if (!is.finite(anchor) || !is.finite(gamma_hat_x) || !is.finite(gamma_hat_2k_x) ||
      !is.finite(rho_hat) || abs(rho_hat) < 1e-10 || abs(denom) < 1e-10) {
    return(NA_real_)
  }
  A_hat <- (1 - rho_hat) * (gamma_hat_x - gamma_hat_2k_x) / denom
  ratio <- k / (n * alpha)
  anchor + gamma_hat_x * log(ratio) + A_hat * (ratio^rho_hat - 1) / rho_hat
}

# LRV estimator for X: since Y = log X is RV, estimate q_Y directly.
lrv_log_weissman <- function(y, k, alpha) {
  y <- sort(y[is.finite(y) & y > 0], decreasing = TRUE)
  n <- length(y)
  if (k < 2 || k >= n) return(NA_real_)
  anchor <- y[k + 1L]
  gamma_hat_y <- hill_estimator(y, k)
  if (!is.finite(anchor) || !is.finite(gamma_hat_y)) return(NA_real_)
  anchor * (k / (n * alpha))^gamma_hat_y
}

lrv_log_corrected_weissman <- function(y, k, alpha, rho_hat) {
  y <- sort(y[is.finite(y) & y > 0], decreasing = TRUE)
  n <- length(y)
  if (k < 2 || k >= n) return(NA_real_)
  anchor <- y[k + 1L]
  gamma_hat_y <- hill_estimator(y, k)
  A_hat <- estimate_A(y, k, rho_hat)
  if (!is.finite(anchor) || !is.finite(gamma_hat_y) || !is.finite(A_hat)) return(NA_real_)
  ratio <- k / (n * alpha)
  # Preserve overflow as Inf rather than truncating the estimate.
  anchor * ratio^gamma_hat_y * exp(A_hat * (ratio^rho_hat - 1) / rho_hat)
}

select_k_tree <- function(values, k_grid) {
  ok <- is.finite(values)
  values <- values[ok]
  grid <- k_grid[ok]
  if (!length(values)) return(k_grid[1L])
  if (length(values) < 4L) return(grid[ceiling(length(grid) / 2)])
  left <- 1L
  right <- length(values)
  while ((right - left) >= 3L) {
    mid <- floor((left + right) / 2)
    v_left <- stats::var(values[left:mid])
    v_right <- stats::var(values[mid:right])
    if (!is.finite(v_left)) v_left <- Inf
    if (!is.finite(v_right)) v_right <- Inf
    if (v_left <= v_right) right <- mid else left <- mid
  }
  grid[floor((left + right) / 2)]
}

select_logq_estimator <- function(y, alpha, k_grid, method, rho_hat) {
  logq_path <- vapply(k_grid, function(k) {
    switch(
      method,
      RV_Weissman = rv_log_weissman(y, k, alpha),
      RV_Corrected = rv_log_corrected_weissman(y, k, alpha, rho_hat),
      LRV_Weissman = lrv_log_weissman(y, k, alpha),
      LRV_Corrected = lrv_log_corrected_weissman(y, k, alpha, rho_hat),
      stop("Unknown method: ", method)
    )
  }, numeric(1))
  selected_k <- select_k_tree(logq_path, k_grid)
  list(logqhat = logq_path[match(selected_k, k_grid)], selected_k = selected_k)
}

compute_logscale_errors <- function(logqhat, logqtrue) {
  if (!is.finite(logqhat) || !is.finite(logqtrue)) {
    return(c(MedSElog = NA_real_, MedSEloglog = NA_real_))
  }
  medse_log <- (logqhat - logqtrue)^2
  medse_loglog <- if (logqhat > 0 && logqtrue > 0) {
    (log(logqhat) - log(logqtrue))^2
  } else {
    NA_real_
  }
  c(
    MedSElog = medse_log,
    MedSEloglog = medse_loglog
  )
}

logq_lrv <- function(p, spec) {
  alpha_tail <- 1 - p
  gamma <- spec$gamma
  rho <- spec$rho
  switch(
    spec$dist,
    Burr = (alpha_tail^rho - 1)^(-gamma / rho),
    NHW = {
      t <- 1 / alpha_tail
      t^gamma * exp((t^rho) * log(t) / 2)
    },
    Frechet = (-log(p))^(-gamma),
    Fisher = stats::qf(p, df1 = 1, df2 = 2 / gamma),
    GPD = if (abs(gamma) < 1e-12) -log(alpha_tail) else (alpha_tail^(-gamma) - 1) / gamma,
    InvGamma = 1 / stats::qgamma(1 - p, shape = 1 / gamma, rate = 1),
    Student = stats::qt((1 + p) / 2, df = 1 / gamma),
    stop("Unknown distribution: ", spec$dist)
  )
}

r_log_lrv <- function(n, spec) {
  u <- stats::runif(n)
  gamma <- spec$gamma
  rho <- spec$rho
  switch(
    spec$dist,
    Burr = ((1 - u)^rho - 1)^(-gamma / rho),
    NHW = {
      t <- 1 / pmax(1 - u, .Machine$double.xmin)
      t^gamma * exp((t^rho) * log(t) / 2)
    },
    Frechet = (-log(u))^(-gamma),
    Fisher = stats::qf(u, df1 = 1, df2 = 2 / gamma),
    GPD = if (abs(gamma) < 1e-12) -log(1 - u) else ((1 - u)^(-gamma) - 1) / gamma,
    InvGamma = 1 / stats::rgamma(n, shape = 1 / gamma, rate = 1),
    Student = abs(stats::rt(n, df = 1 / gamma)),
    stop("Unknown distribution: ", spec$dist)
  )
}

make_logscale_specs <- function() {
  specs <- list()
  add <- function(name, dist, gamma, rho) {
    specs[[length(specs) + 1L]] <<- list(name = name, dist = dist, gamma = gamma, rho = rho)
  }
   for (g in c(1/8, 1/4, 1/2, 1)) add(sprintf("Burr_g%g_r-0.125", g), "Burr", g, -1/8)
   for (r in c(-1/4, -1/2, -1, -2)) add(sprintf("Burr_g1_r%g", r), "Burr", 1, r)
   for (g in c(1/8, 1/4, 1/2, 1)) add(sprintf("NHW_g%g_r-0.125", g), "NHW", g, -1/8)
   for (r in c(-1/4, -1/2, -1, -2)) add(sprintf("NHW_g1_r%g", r), "NHW", 1, r)
   add("Fisher_g0.125_r-0.125", "Fisher", 1/8, -1/8)
   add("Fisher_g1_r-1", "Fisher", 1, -1)
   add("GPD_g0.125_r-0.125", "GPD", 1/8, -1/8)
   add("InvGamma_g1_r-1", "InvGamma", 1, -1)
   add("Student_g1_r-2", "Student", 1, -2)
   #add("Burr_g1_r-0.125", "Burr", 1, -1/8)
   #add("Student_g1_r-2", "Student", 1, -2)
  specs
}
