#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("data.table", quietly = TRUE)) {
    stop("R package 'data.table' is required")
  }
  if (!requireNamespace("WGCNA", quietly = TRUE)) {
    stop("R package 'WGCNA' is required")
  }
  library(data.table)
  library(WGCNA)
})

parse_arguments <- function(arguments) {
  defaults <- list(
    matrix = NULL,
    output_dir = NULL,
    seed = 42L,
    threads = 1L,
    scale_free_r2 = 0.8,
    min_mean_connectivity = 5.0,
    deep_split = 2L,
    min_module_size = 30L,
    merge_cut_height = 0.25,
    max_block_size = 5000L,
    max_p_outliers = 0.1
  )
  index <- 1L
  while (index <= length(arguments)) {
    key <- arguments[[index]]
    if (!startsWith(key, "--") || index == length(arguments)) {
      stop(sprintf("Invalid argument sequence near: %s", key))
    }
    value <- arguments[[index + 1L]]
    normalized <- gsub("-", "_", substring(key, 3L))
    if (!(normalized %in% names(defaults))) {
      stop(sprintf("Unknown argument: %s", key))
    }
    defaults[[normalized]] <- value
    index <- index + 2L
  }
  defaults$seed <- as.integer(defaults$seed)
  defaults$threads <- as.integer(defaults$threads)
  defaults$scale_free_r2 <- as.numeric(defaults$scale_free_r2)
  defaults$min_mean_connectivity <- as.numeric(defaults$min_mean_connectivity)
  defaults$deep_split <- as.integer(defaults$deep_split)
  defaults$min_module_size <- as.integer(defaults$min_module_size)
  defaults$merge_cut_height <- as.numeric(defaults$merge_cut_height)
  defaults$max_block_size <- as.integer(defaults$max_block_size)
  defaults$max_p_outliers <- as.numeric(defaults$max_p_outliers)
  defaults
}

arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))
if (is.null(arguments$matrix) || is.null(arguments$output_dir)) {
  stop("--matrix and --output-dir are required")
}
if (!file.exists(arguments$matrix)) {
  stop(sprintf("Replicate-fit matrix does not exist: %s", arguments$matrix))
}
if (arguments$threads < 1L) {
  stop("--threads must be positive")
}
if (arguments$scale_free_r2 <= 0 || arguments$scale_free_r2 > 1) {
  stop("--scale-free-r2 must be in (0, 1]")
}

dir.create(arguments$output_dir, recursive = TRUE, showWarnings = FALSE)
if (arguments$threads > 1L) {
  allowWGCNAThreads(nThreads = arguments$threads)
} else {
  disableWGCNAThreads()
}
set.seed(arguments$seed)

read_command <- if (endsWith(arguments$matrix, ".gz")) {
  sprintf("gzip -cd %s", shQuote(normalizePath(arguments$matrix)))
} else {
  NULL
}
fit_table <- if (is.null(read_command)) {
  fread(arguments$matrix, sep = "\t", header = TRUE, check.names = FALSE)
} else {
  fread(cmd = read_command, sep = "\t", header = TRUE, check.names = FALSE)
}
if (ncol(fit_table) < 3L || names(fit_table)[[1L]] != "sample_id") {
  stop("Replicate-fit matrix must start with sample_id and contain at least two Genes")
}
sample_ids <- as.character(fit_table[[1L]])
if (anyDuplicated(sample_ids)) {
  stop("Replicate-fit matrix contains duplicate sample IDs")
}
gene_ids <- names(fit_table)[-1L]
if (anyDuplicated(gene_ids)) {
  stop("Replicate-fit matrix contains duplicate Gene IDs")
}
dat_expr <- as.data.frame(fit_table[, -1L], check.names = FALSE)
numeric_columns <- vapply(dat_expr, is.numeric, logical(1))
if (!all(numeric_columns)) {
  stop(sprintf(
    "Replicate-fit matrix contains non-numeric Gene columns: %s",
    paste(head(gene_ids[!numeric_columns], 10L), collapse = ",")
  ))
}
dat_expr[] <- lapply(dat_expr, as.numeric)
names(dat_expr) <- gene_ids
rownames(dat_expr) <- sample_ids
if (any(!is.finite(as.matrix(dat_expr)))) {
  stop("Replicate-fit matrix contains non-finite values")
}
quality <- goodSamplesGenes(dat_expr, verbose = 0)
if (!quality$allOK) {
  bad_genes <- gene_ids[!quality$goodGenes]
  bad_samples <- sample_ids[!quality$goodSamples]
  stop(sprintf(
    "WGCNA goodSamplesGenes failed: Genes=%s Samples=%s",
    paste(head(bad_genes, 10L), collapse = ","),
    paste(head(bad_samples, 10L), collapse = ",")
  ))
}

powers <- 1:50
soft_threshold <- pickSoftThreshold(
  dat_expr,
  powerVector = powers,
  networkType = "signed",
  corFnc = "bicor",
  corOptions = list(use = "p", maxPOutliers = arguments$max_p_outliers),
  verbose = 2
)
fit_indices <- as.data.table(soft_threshold$fitIndices)
setnames(fit_indices, make.names(names(fit_indices), unique = TRUE))
fwrite(
  fit_indices,
  file.path(arguments$output_dir, "soft_threshold_diagnostics.tsv"),
  sep = "\t"
)
required_fit_columns <- c("Power", "SFT.R.sq", "mean.k.")
if (!all(required_fit_columns %in% names(fit_indices))) {
  stop(sprintf(
    "Unexpected pickSoftThreshold columns: %s",
    paste(names(fit_indices), collapse = ",")
  ))
}
eligible_power <- fit_indices[
  SFT.R.sq >= arguments$scale_free_r2 & mean.k. >= arguments$min_mean_connectivity
]
if (nrow(eligible_power) == 0L) {
  stop(sprintf(
    "No power 1-50 met SFT.R.sq >= %.3f and mean connectivity >= %.3f",
    arguments$scale_free_r2,
    arguments$min_mean_connectivity
  ))
}
selected_power <- as.integer(eligible_power$Power[[1L]])

network <- blockwiseModules(
  dat_expr,
  power = selected_power,
  networkType = "signed",
  TOMType = "signed",
  corType = "bicor",
  maxPOutliers = arguments$max_p_outliers,
  deepSplit = arguments$deep_split,
  minModuleSize = arguments$min_module_size,
  mergeCutHeight = arguments$merge_cut_height,
  maxBlockSize = arguments$max_block_size,
  numericLabels = TRUE,
  pamRespectsDendro = FALSE,
  reassignThreshold = 0,
  randomSeed = arguments$seed,
  saveTOMs = FALSE,
  verbose = 3
)
module_labels <- as.integer(network$colors)
if (length(module_labels) != length(gene_ids)) {
  stop("WGCNA module axis does not match Gene axis")
}
non_grey_labels <- sort(unique(module_labels[module_labels != 0L]))
if (length(non_grey_labels) == 0L) {
  stop("WGCNA returned no non-grey modules")
}
module_id_map <- setNames(
  sprintf("M%03d", seq_along(non_grey_labels)),
  as.character(non_grey_labels)
)
module_ids <- ifelse(
  module_labels == 0L,
  "grey",
  unname(module_id_map[as.character(module_labels)])
)
module_colors <- labels2colors(module_labels)
eigengene_result <- moduleEigengenes(
  dat_expr,
  colors = module_labels,
  excludeGrey = TRUE,
  grey = 0,
  scale = TRUE
)
module_eigengenes <- orderMEs(eigengene_result$eigengenes)
kme <- signedKME(dat_expr, module_eigengenes, outputColumnName = "kME")
assigned_kme <- rep(NA_real_, length(gene_ids))
for (gene_index in seq_along(gene_ids)) {
  label <- module_labels[[gene_index]]
  if (label == 0L) {
    next
  }
  matching_columns <- grep(
    sprintf("^kME%s$", label),
    colnames(kme),
    value = TRUE
  )
  if (length(matching_columns) == 1L) {
    assigned_kme[[gene_index]] <- kme[gene_index, matching_columns]
  }
}
gene_modules <- data.table(
  gene_id = gene_ids,
  module_id = module_ids,
  wgcna_numeric_label = module_labels,
  wgcna_color = module_colors,
  assigned_kme = assigned_kme
)
fwrite(gene_modules, file.path(arguments$output_dir, "gene_modules.tsv"), sep = "\t")

eigengene_table <- data.table(sample_id = sample_ids)
for (column_name in colnames(module_eigengenes)) {
  numeric_label <- sub("^ME", "", column_name)
  module_id <- unname(module_id_map[numeric_label])
  if (length(module_id) == 1L && !is.na(module_id)) {
    eigengene_table[[module_id]] <- module_eigengenes[[column_name]]
  }
}
fwrite(
  eigengene_table,
  file.path(arguments$output_dir, "wgcna_module_eigengenes.tsv"),
  sep = "\t"
)

parameter_table <- data.table(
  parameter = c(
    "selected_power",
    "network_type",
    "correlation",
    "scale_free_r2_threshold",
    "min_mean_connectivity",
    "deep_split",
    "min_module_size",
    "merge_cut_height",
    "max_block_size",
    "max_p_outliers",
    "seed",
    "threads",
    "n_samples",
    "n_genes",
    "n_non_grey_modules",
    "r_version",
    "wgcna_version",
    "data_table_version"
  ),
  value = as.character(c(
    selected_power,
    "signed",
    "bicor",
    arguments$scale_free_r2,
    arguments$min_mean_connectivity,
    arguments$deep_split,
    arguments$min_module_size,
    arguments$merge_cut_height,
    arguments$max_block_size,
    arguments$max_p_outliers,
    arguments$seed,
    arguments$threads,
    nrow(dat_expr),
    ncol(dat_expr),
    length(non_grey_labels),
    R.version.string,
    as.character(packageVersion("WGCNA")),
    as.character(packageVersion("data.table"))
  ))
)
fwrite(parameter_table, file.path(arguments$output_dir, "wgcna_parameters.tsv"), sep = "\t")
saveRDS(
  list(
    network = network,
    selected_power = selected_power,
    gene_ids = gene_ids,
    sample_ids = sample_ids,
    module_id_map = module_id_map,
    parameters = arguments
  ),
  file.path(arguments$output_dir, "wgcna_model.rds"),
  compress = "xz"
)
cat(sprintf(
  "WGCNA complete: power=%d Genes=%d modules=%d\n",
  selected_power,
  length(gene_ids),
  length(non_grey_labels)
))
