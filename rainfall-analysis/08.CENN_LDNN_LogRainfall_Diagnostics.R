rm(list = ls())

# =============================================================================
# Conditional CENN/LDNN log-rainfall diagnostics and publication plots
# =============================================================================
# This script reads the outputs from:
#   07.Conditional_CENN_LDNN_LogRainfall.R
#
# Publication figures are saved to RUN_DIR, and the plot objects remain
# available in the workspace for further inspection.
# =============================================================================

# ---- 0. Packages -------------------------------------------------------------
library(data.table)
library(ggplot2)
library(viridis)
library(scales)

# ---- 1. User-editable settings ----------------------------------------------
script_path <- tryCatch(
  normalizePath(sys.frame(1)$ofile, winslash = "/"),
  error = function(e) NA_character_
)
if (is.na(script_path) || !nzchar(script_path)) {
  file_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(file_arg)) {
    script_path <- normalizePath(sub("^--file=", "", file_arg[1L]), winslash = "/")
  } else {
    script_path <- normalizePath("08.CENN_LDNN_LogRainfall_Diagnostics.R", winslash = "/")
  }
}

SCRIPT_DIR <- dirname(script_path)
BASE_DIR <- normalizePath(
  Sys.getenv("CENN_LDNN_BASE_DIR", unset = SCRIPT_DIR),
  winslash = "/",
  mustWork = FALSE
)
RUN_DIR <- Sys.getenv(
  "CENN_LDNN_RUN_DIR",
  unset = file.path(BASE_DIR, "outputs", "cenn_ldnn_lograinfall")
)
COORD_FILE <- Sys.getenv(
  "CENN_LDNN_COORD_FILE",
  unset = file.path(BASE_DIR, "data", "western_ghats_coordinates.csv")
)
TARGET_FILE <- file.path(RUN_DIR, "sampled_target_grids.csv")
SELECTED_FILE <- file.path(RUN_DIR, "all_selected_models.csv")
CANDIDATE_FILE <- file.path(RUN_DIR, "all_candidate_models.csv")

#Neighbourhood count used in diagnostics
DIAG_NEIGHBOR_COUNT <- NA_integer_    # NA selects the first available value.

# Model displayed in spatial diagnostics.
BEST_MODEL <- "CENN-LRV"

# Tail diagnostic controls.
MAX_HILL_K <- 250L
SPACING_K <- 100L

# Candidate-path controls.
PATH_MODELS <- c("CENN-LRV", "LDNN-LRV", "CENN-RV", "LDNN-RV")

REGION_LEVELS <- c("coastal", "mid_range", "hilly")
REGION_LABELS <- c(
  coastal = "Coastal",
  mid_range = "Mid-range",
  hilly = "Hilly"
)

REGION_COLORS <- c(
  Coastal = "#0072B2",
  `Mid-range` = "#D55E00",
  Hilly = "#009E73"
)

# ---- 2. Read analysis outputs ------------------------------------------------
coords <- fread(COORD_FILE)
targets <- fread(TARGET_FILE)
selected <- fread(SELECTED_FILE)
candidates <- fread(CANDIDATE_FILE)

targets[, region := factor(region, levels = REGION_LEVELS, labels = REGION_LABELS)]
selected[, target_region := factor(target_region, levels = REGION_LEVELS, labels = REGION_LABELS)]
candidates[, target_region := factor(target_region, levels = REGION_LEVELS, labels = REGION_LABELS)]

available_m <- sort(unique(selected$neighbor_count))
if (is.na(DIAG_NEIGHBOR_COUNT)) DIAG_NEIGHBOR_COUNT <- available_m[1L]

selected_m <- selected[neighbor_count == DIAG_NEIGHBOR_COUNT]
candidates_m <- candidates[neighbor_count == DIAG_NEIGHBOR_COUNT]

target_lookup <- unique(selected_m[, .(
  target_grid_id,
  target_lat,
  target_lon,
  target_region,
  target_alt_mean,
  heldout_log_max,
  heldout_rainfall
)])

global_input_file <- file.path(
  RUN_DIR,
  "inputs",
  paste0("global_m", DIAG_NEIGHBOR_COUNT, "_input.csv")
)
global_input <- fread(global_input_file)

# ---- 3. Common plotting theme ------------------------------------------------
theme_paper <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "bottom",
      strip.text = element_text(face = "bold"),
      plot.title = element_text(face = "bold"),
      plot.subtitle = element_text(color = "grey35"),
      axis.title = element_text(face = "bold")
    )
}

# ---- 4. Tail-scale diagnostic helpers ---------------------------------------
# RV curve: Hill-type spacing on Y = log(rainfall).
# LRV curve: Hill-type spacing on T = log(Y) = log(log(rainfall)).
hill_curve <- function(values, scale_name, max_k = MAX_HILL_K) {
  values <- values[is.finite(values)]
  values <- sort(values, decreasing = TRUE)
  if (length(values) < 5L) return(data.table())

  max_k <- min(max_k, length(values) - 1L)
  ks <- seq_len(max_k)
  data.table(
    scale = scale_name,
    k = ks,
    hill = vapply(ks, function(k) mean(values[seq_len(k)]) - values[k + 1L], numeric(1L))
  )
}

make_target_hill_curves <- function(input_dt, target_info, max_k = MAX_HILL_K) {
  paths <- lapply(seq_len(nrow(target_info)), function(i) {
    grid <- target_info$target_grid_id[i]
    y <- input_dt[grid_id == grid, Y]
    rv <- hill_curve(y, "RV: log-rainfall scale", max_k)
    lrv <- hill_curve(log(y[y > 0]), "LRV: double-log scale", max_k)
    out <- rbindlist(list(rv, lrv), fill = TRUE)
    if (!nrow(out)) return(NULL)
    out[, `:=`(
      target_grid_id = grid,
      target_region = target_info$target_region[i],
      target_alt_mean = target_info$target_alt_mean[i]
    )]
    out
  })
  rbindlist(paths, fill = TRUE)
}

spacing_data_one <- function(y, scale_name, k = SPACING_K) {
  y <- y[is.finite(y)]
  y <- sort(y, decreasing = TRUE)
  if (length(y) <= k || k < 3L) return(data.table())

  i <- seq_len(k - 1L)
  data.table(
    scale = scale_name,
    i = i,
    k = k,
    x_log_k_over_i = log(k / i),
    empirical_spacing = y[i] - y[k]
  )
}

make_spacing_diagnostics <- function(input_dt, target_info, k = SPACING_K) {
  paths <- lapply(seq_len(nrow(target_info)), function(i) {
    grid <- target_info$target_grid_id[i]
    y <- input_dt[grid_id == grid, Y]
    rv <- spacing_data_one(y, "RV: log-rainfall scale", k)
    lrv <- spacing_data_one(log(y[y > 0]), "LRV: double-log scale", k)
    out <- rbindlist(list(rv, lrv), fill = TRUE)
    if (!nrow(out)) return(NULL)
    out[, `:=`(
      target_grid_id = grid,
      target_region = target_info$target_region[i],
      target_alt_mean = target_info$target_alt_mean[i]
    )]
    out
  })
  rbindlist(paths, fill = TRUE)
}

choose_representative_targets <- function(target_info) {
  # One target per elevation group, chosen near the median altitude in that group.
  reps <- target_info[
    ,
    {
      med_alt <- median(target_alt_mean, na.rm = TRUE)
      .SD[which.min(abs(target_alt_mean - med_alt))][1L]
    },
    by = target_region
  ]
  reps[order(target_region)]
}

# ---- 5. Figure 1: Study region and sampled target grids ----------------------
fig_01_study_region <- ggplot(coords, aes(x = lon, y = lat, fill = alt_mean)) +
  geom_tile() +
  # Target grid points layer
  geom_point(
    data = targets,
    aes(x = lon, y = lat, color = region),
    inherit.aes = FALSE,
    size = 2.5,
    alpha = 0.95
  ) +
  # Geographic text annotation for Arabian Sea
  annotate(
    "text", x = 74.5, y = 11.5, 
    label = "ARABIAN SEA", 
    color = "#1C5480", 
    fontface = "italic", 
    size = 4.5, 
    hjust = 0.5
  ) +
 # Compass marker.
  annotate(
    "text", x = 74.5, y = 9.8, 
    label = "N\nW <     > E\nS", 
    color = "grey20", 
    fontface = "bold", 
    lineheight = 0.9,
    size = 3.5, 
    hjust = 0.5
  ) +
 
  coord_fixed(clip = "off") +
  scale_fill_viridis_c(
    name = "Mean altitude (m)", 
    option = "D",
    guide = guide_colorbar(barheight = unit(0.3, "cm"), barwidth = unit(5, "cm"))
  ) +
  scale_color_manual(name = NULL, values = REGION_COLORS, drop = FALSE) +
  labs(
    #title = paste0("Study Region and Sampled Target Grids (n = ", nrow(targets), ")"),
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_paper() +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",             
    legend.box.just = "center",
    legend.margin = margin(t = -5, r = 0, b = 0, l = 0), 
    legend.spacing.y = unit(0.15, "cm"), 
    legend.box.spacing = unit(0.2, "cm"),
    plot.margin = margin(t = 15, r = 15, b = 15, l = 25, unit = "pt")
  )

print(fig_01_study_region)
ggsave(
  filename = file.path(RUN_DIR, "fig_01_study_region.pdf"),
  plot = fig_01_study_region,
  width = 5.5,         
  height = 7.0,        
  units = "in",
  device = cairo_pdf   
)

# ---- 6. Figure 2a: Representative RV/LRV Hill-type stability plots -----------
representative_targets <- choose_representative_targets(target_lookup)
hill_rep <- make_target_hill_curves(global_input, representative_targets, MAX_HILL_K)

fig_02a_representative_hill <- ggplot(hill_rep, aes(x = k, y = hill, color = scale)) +
  geom_line(linewidth = 0.8) +
  facet_wrap(~ target_region, scales = "free_y") +
  scale_color_manual(
    name = "Tail scale",
    values = c(
      "RV: log-rainfall scale" = "#0072B2",
      "LRV: double-log scale" = "#D55E00"
    )
  ) +
  labs(
    #title = "RV and LRV Hill-Type Stability for Representative Targets",
    #subtitle = paste0("One target per elevation group; neighbour count = ", DIAG_NEIGHBOR_COUNT),
    x = "Upper order threshold k",
    y = "Hill-type transformed spacing"
  ) +
  theme_paper()

print(fig_02a_representative_hill)

ggsave(
  filename = file.path(RUN_DIR, "fig_02a_representative_hill.pdf"),
  plot = fig_02a_representative_hill,
  width = 8,          
  height = 4,         
  units = "in",
  device = cairo_pdf 
)

# ---- 7. Figure 2b: All target Hill curves by elevation group -----------------
hill_all <- make_target_hill_curves(target_info = target_lookup, input_dt = global_input, max_k = MAX_HILL_K)

fig_02b_all_hill_curves <- ggplot(
  hill_all,
  aes(x = k, y = hill, group = target_grid_id, color = target_region)
) +
  geom_line(alpha = 0.22, linewidth = 0.35) +
  stat_summary(aes(group = 1), fun = median, geom = "line", color = "black", linewidth = 0.9) +
  facet_grid(target_region ~ scale, scales = "free_y") +
  scale_color_manual(name = "Elevation group", values = REGION_COLORS, drop = FALSE) +
  labs(
    title = "Hill-Type Stability Curves for All Sampled Targets",
    subtitle = "Thin lines are target-grid curves; black line is the group median",
    x = "Upper order threshold k",
    y = "Hill-type transformed spacing"
  ) +
  theme_paper()

print(fig_02b_all_hill_curves)

ggsave(
  filename = file.path(RUN_DIR, "fig_02b_all_hill_curves.pdf"),
  plot = fig_02b_all_hill_curves,
  width = 8,          
  height = 4,         
  units = "in",
  device = cairo_pdf 
)

# ---- 8. Figure 2c: Spacing linearity diagnostics -----------------------------

make_spacing_diagnostics <- function(input_dt, target_info, k = SPACING_K) {
  paths <- lapply(seq_len(nrow(target_info)), function(loop_idx) { 
    grid <- target_info$target_grid_id[loop_idx]
    
    current_region <- target_info$target_region[loop_idx]
    current_alt    <- target_info$target_alt_mean[loop_idx]
    
    y <- input_dt[grid_id == grid, Y]
    rv <- spacing_data_one(y, "RV: log-rainfall scale", k)
    lrv <- spacing_data_one(log(y[y > 0]), "LRV: double-log scale", k)
    
    out <- rbindlist(list(rv, lrv), fill = TRUE)
    if (!nrow(out)) return(NULL)
    
    out[, `:=`(
      target_grid_id = grid,
      target_region = current_region,
      target_alt_mean = current_alt
    )]
    out
  })
  rbindlist(paths, fill = TRUE)
}
spacing_rep <- make_spacing_diagnostics(global_input, representative_targets, SPACING_K)

fig_02c_spacing_linearity <- ggplot(
  spacing_rep,
  aes(x = x_log_k_over_i, y = empirical_spacing, color = scale)
) +
  geom_point(size = 1.1, alpha = 0.70) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  facet_wrap(~ target_region, scales = "free_y") +
  scale_color_manual(
    name = "Tail scale",
    values = c(
      "RV: log-rainfall scale" = "#0072B2",
      "LRV: double-log scale" = "#D55E00"
    )
  ) +
  labs(
    #title = "Upper-Order Spacing Linearity",
    #subtitle = paste0("Spacing threshold k = ", SPACING_K),
    x = "log(k / i)",
    y = "Empirical transformed spacing"
  ) +
  theme_paper()

print(fig_02c_spacing_linearity)

ggsave(
  filename = file.path(RUN_DIR, "fig_02c_spacing_linearity.pdf"),
  plot = fig_02c_spacing_linearity,
  width = 10.0,         
  height = 5.5,         
  units = "in",
  device = cairo_pdf   
)


# ---- 9. Figure 3: Model comparison by elevation group ------------------------
selected_plot <- selected_m[is.finite(MSE_log)]
selected_plot[, model := factor(model, levels = intersect(PATH_MODELS, unique(model)))]

fig_03_model_comparison <- ggplot(
  selected_plot,
  aes(x = model, y = MSE_log, fill = target_region)
) +
  geom_boxplot(width = 0.65, outlier.alpha = 0.45) +
  scale_y_continuous(trans = "log10", labels = label_number()) +
  scale_fill_manual(name = "Elevation group", values = REGION_COLORS, drop = FALSE) +
  labs(
    #title = "Held-Out Squared Error by Model and Elevation Group",
    #subtitle = paste0("Error scale: squared error for held-out local maximum; neighbour count = ", DIAG_NEIGHBOR_COUNT),
    #x = "Model",
    y = expression(SE[log])
  ) +
  theme_paper()

print(fig_03_model_comparison)

ggsave(
  filename = file.path(RUN_DIR, "fig_03_model_comparison.pdf"),
  plot = fig_03_model_comparison,
  width = 8,          
  height = 4,         
  units = "in",
  device = cairo_pdf
)

# ---- 10. Figure 4: SE_log Performance Maps across Estimators -----------------

target_models <- c("CENN-LRV", "CENN-RV", "LDNN-LRV", "LDNN-RV")

# Cap the colour scale at the 95th percentile.
valid_mse <- selected_m[model %in% target_models & is.finite(MSE_log) & MSE_log < 1e5]
global_min_mse <- min(valid_mse$MSE_log, na.rm = TRUE)
global_max_mse <- quantile(valid_mse$MSE_log, probs = 0.95, na.rm = TRUE) 

SHAPE_MAPPING <- c("Coastal" = 21, "Mid-range" = 22, "Hilly" = 24)

for (current_model in target_models) {
  
  model_estimates <- selected_m[model == current_model & is.finite(MSE_log)]
  if (nrow(model_estimates) == 0) next
  
  plot_data <- merge(model_estimates, targets, by.x = c("target_lon", "target_lat"), by.y = c("lon", "lat"))
  plot_data[MSE_log > global_max_mse, MSE_log := global_max_mse]
  
  p_map <- ggplot() +
    # LAYER 1: Base continuous terrain elevation map
    geom_tile(
      data = coords, 
      aes(x = lon, y = lat, fill = alt_mean),
      inherit.aes = FALSE
    ) +
    scale_fill_viridis_c(
      name = expression(Altitude~(m)), 
      option = "D",
      guide = guide_colorbar(
        title.position = "left",  
        title.vjust = 0.5,        
        barheight = unit(0.2, "cm"), 
        barwidth = unit(3.2, "cm")   
      )
    ) +
    
    # BIND NEW SCALES
    ggnewscale::new_scale_color() +
    ggnewscale::new_scale_fill() +
    
    # LAYER 2: Gauge points mapping terrain (Shape) and MSE_log (Fill)
    geom_point(
      data = plot_data,
      aes(x = target_lon, y = target_lat, fill = MSE_log, shape = region),
      color = "#FFFFFF",  
      stroke = 0.4,
      size = 3.2,
      alpha = 0.95
    ) +
    
  
    annotate(
      "text", x = 74.5, y = 11.5, 
      label = "ARABIAN SEA", 
      color = "#1C5480", 
      fontface = "italic", 
      size = 4.5, 
      hjust = 0.5
    ) +
    
    annotate(
      "text", x = 74.5, y = 9.8, 
      label = "N\nW <     > E\nS", 
      color = "grey20", 
      fontface = "bold", 
      lineheight = 0.9,
      size = 3.5, 
      hjust = 0.5
    ) +
    
    coord_fixed(clip = "off") + 
    
    # HORIZONTAL SCALE: For Capped SE_log
    scale_fill_viridis_c(
      name = expression(SE[log]),  
      option = "A", 
      limits = c(global_min_mse, global_max_mse),
      labels = function(x) sprintf("%.1f", x), 
      guide = guide_colorbar(
        title.position = "left",  
        title.vjust = 0.5,        
        barheight = unit(0.2, "cm"),  
        barwidth = unit(3.2, "cm")    
      )
    ) +
    
    scale_shape_manual(
      name = NULL,
      values = SHAPE_MAPPING,
      drop = FALSE,
      guide = guide_legend(
        nrow = 1,
        override.aes = list(size = 2.8, fill = "grey50", color = "black")
      )
    ) +
    
    labs(x = "Longitude", y = "Latitude") +
    theme_paper() +
    theme(
      legend.position = "bottom",
      legend.box = "vertical",             
      legend.box.just = "center",
      legend.margin = margin(t = -5, r = 0, b = 0, l = 0), 
      legend.spacing.x = unit(0.6, "cm"),   
      legend.spacing.y = unit(0.15, "cm"),  
      legend.box.spacing = unit(0.2, "cm"),
      
      legend.title = element_text(size = 7.5, face = "bold"),
      legend.text = element_text(size = 7.0),
      
      plot.margin = margin(t = 15, r = 15, b = 15, l = 25, unit = "pt")
    )
  
  clean_model_name <- tolower(gsub("-", "_", current_model))
  ggsave(
    filename = file.path(RUN_DIR, paste0("fig_04_mse_", clean_model_name, ".pdf")),
    plot = p_map, width = 5.8, height = 6.8, units = "in", device = cairo_pdf   
  )
}

# ---- 11. Candidate-path stability plots for representative targets -----------
candidates_plot <- merge(
  candidates_m,
  target_lookup[, .(target_grid_id, target_lat, target_lon, target_alt_mean)],
  by = "target_grid_id",
  all.x = TRUE
)
candidates_plot <- candidates_plot[
  target_grid_id %in% representative_targets$target_grid_id &
    model %in% PATH_MODELS &
    is.finite(logqhat)
]
candidates_plot[, model_path := fifelse(
  model_family == "LDNN",
  paste0(model, ", j=", k_tau),
  model
)]

fig_05_quantile_path <- ggplot(
  candidates_plot,
  aes(x = k_delta, y = logqhat, color = model_path, group = model_path)
) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 1.4) +
  facet_wrap(~ target_region, scales = "free_y") +
  labs(
    title = "Candidate Extrapolation Paths over Anchor k",
    subtitle = "Representative targets; selected models are chosen by minimum anchor-MAD",
    x = "Anchor order statistic k",
    y = "Estimated extreme log-rainfall quantile",
    color = "Candidate path"
  ) +
  theme_paper()

print(fig_05_quantile_path)

# ---- 12. Map of held-out local maxima ----------------------------------------

plot_truth <- merge(
  unique(selected_m[, .(target_lon, target_lat, heldout_log_max)]), 
  targets, 
  by.x = c("target_lon", "target_lat"), 
  by.y = c("lon", "lat")
)

p_truth <- ggplot() +
  geom_tile(
    data = coords, 
    aes(x = lon, y = lat, fill = alt_mean),
    inherit.aes = FALSE
  ) +
  scale_fill_viridis_c(
    name = "Mean altitude (m)", 
    option = "D",
    guide = guide_colorbar(
      title.position = "left",  
      title.vjust = 0.5,        
      barheight = unit(0.3, "cm"), 
      barwidth = unit(4.0, "cm")
    )
  ) +
  
  ggnewscale::new_scale_color() +
  ggnewscale::new_scale_fill() +
  
  # target grid circles colored by their held-out local maximum
  geom_point(
    data = plot_truth,
    aes(x = target_lon, y = target_lat, fill = heldout_log_max),
    shape = 21,       
    color = "#FFFFFF", 
    stroke = 0.5,
    size = 3.5,
    alpha = 0.95
  ) +
  
  annotate(
    "text", x = 74.5, y = 11.5, 
    label = "ARABIAN SEA", 
    color = "#1C5480", 
    fontface = "italic", 
    size = 4.5, 
    hjust = 0.5
  ) +
  
  annotate(
    "text", x = 74.5, y = 9.8, 
    label = "N\nW <     > E\nS", 
    color = "grey20", 
    fontface = "bold", 
    lineheight = 0.9,
    size = 3.5, 
    hjust = 0.5
  ) +
  
  coord_fixed(clip = "off") + 
  

  scale_fill_viridis_c(
    name = "Held-out local\nmaximum, log scale", 
    option = "B", 
    guide = guide_colorbar(
      title.position = "left",  
      title.vjust = 0.5,        
      barheight = unit(0.3, "cm"),  
      barwidth = unit(3.7, "cm")    
    )
  ) +
  
  labs(x = "Longitude", y = "Latitude") +
  theme_paper() +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",             
    legend.box.just = "center",
    legend.margin = margin(t = -5, r = 0, b = 0, l = 0), 
    legend.spacing.x = unit(0.6, "cm"),   
    legend.spacing.y = unit(0.15, "cm"),  
    legend.box.spacing = unit(0.2, "cm"),
    
    legend.title = element_text(size = 7.5, face = "bold"),
    legend.text = element_text(size = 7.0),
    
    plot.margin = margin(t = 15, r = 15, b = 15, l = 25, unit = "pt")
  )

# ---- 13. Altitude versus estimated extreme log-rainfall across Estimators ---

target_models <- c("CENN-LRV", "CENN-RV", "LDNN-LRV", "LDNN-RV")

for (current_model in target_models) {
  
  model_estimates <- selected_m[model == current_model & is.finite(logqhat)]
  if (nrow(model_estimates) == 0) next

  plot_data <- merge(model_estimates, targets, by.x = c("target_lon", "target_lat"), by.y = c("lon", "lat"))
  
  p_trend <- ggplot(
    plot_data,
    aes(x = target_alt_mean, y = logqhat, color = target_region)
  ) +
    geom_point(size = 2.2, alpha = 0.90) +
    geom_smooth(method = "loess", se = TRUE, color = "black", linewidth = 0.8) +
    scale_color_manual(name = NULL, values = REGION_COLORS, drop = FALSE) +
    labs(
      x = "Mean altitude (m)",
      y = "Estimated extreme log-rainfall quantile"
    ) +
    theme_paper() +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box.just = "center",
      legend.margin = margin(t = -5, r = 0, b = 0, l = 0), 
      legend.box.spacing = unit(0.2, "cm"),
      
      legend.text = element_text(size = 7.5),
      axis.title = element_text(size = 8.5, face = "bold"),
      axis.text = element_text(size = 7.5),
      
      plot.margin = margin(t = 15, r = 20, b = 15, l = 25, unit = "pt")
    )
  
  clean_model_name <- tolower(gsub("-", "_", current_model))
  ggsave(
    filename = file.path(RUN_DIR, paste0("fig_07_alt_vs_est_", clean_model_name, ".pdf")),
    plot = p_trend, width = 5.8, height = 4.2, units = "in", device = cairo_pdf   
  )
}

