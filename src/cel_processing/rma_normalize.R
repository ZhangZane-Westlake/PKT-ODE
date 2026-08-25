#!/usr/bin/env Rscript

# Perform organ-wise RMA over all retained Open TG-GATEs rat time points.

suppressPackageStartupMessages({
  library(affy)
  library(Biobase)
  library(data.table)
  library(makecdfenv)
})

Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  BLIS_NUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

parse_arguments <- function(arguments) {
  values <- list(
    organ = "all",
    project_root = NULL,
    manifest = NULL,
    output_dir = NULL
  )
  index <- 1L
  while (index <= length(arguments)) {
    flag <- arguments[[index]]
    if (!(flag %in% c("--organ", "--project-root", "--manifest", "--output-dir"))) {
      stop(sprintf("Unknown argument: %s", flag))
    }
    if (index == length(arguments)) {
      stop(sprintf("Missing value for %s", flag))
    }
    key <- sub("^--", "", flag)
    key <- gsub("-", "_", key)
    values[[key]] <- arguments[[index + 1L]]
    index <- index + 2L
  }
  values
}

detect_project_root <- function() {
  command_arguments <- commandArgs(trailingOnly = FALSE)
  file_argument <- command_arguments[grepl("^--file=", command_arguments)]
  if (length(file_argument) != 1L) {
    stop("Cannot determine script path; pass --project-root explicitly")
  }
  script_path <- normalizePath(sub("^--file=", "", file_argument[[1L]]))
  normalizePath(file.path(dirname(script_path), "../.."))
}

parse_manifest_boolean <- function(values, column_name) {
  normalized <- tolower(trimws(as.character(values)))
  parsed <- rep(NA, length(normalized))
  parsed[normalized %in% c("true", "1")] <- TRUE
  parsed[normalized %in% c("false", "0")] <- FALSE
  if (anyNA(parsed)) {
    invalid <- unique(normalized[is.na(parsed)])
    stop(sprintf("Invalid boolean values in %s: %s", column_name, paste(invalid, collapse = ", ")))
  }
  parsed
}

resolve_cel_paths <- function(paths, project_root) {
  is_absolute <- grepl("^/", paths)
  resolved <- paths
  resolved[!is_absolute] <- file.path(project_root, paths[!is_absolute])
  normalizePath(resolved, mustWork = TRUE)
}

write_expression_matrix <- function(expression_matrix, output_path) {
  output_table <- as.data.table(expression_matrix, keep.rownames = "gene_id")
  temporary_path <- paste0(output_path, ".tmp")
  on.exit(unlink(temporary_path), add = TRUE)
  fwrite(
    output_table,
    file = temporary_path,
    sep = "\t",
    quote = FALSE,
    na = "NA",
    compress = "gzip"
  )
  if (!file.rename(temporary_path, output_path)) {
    stop(sprintf("Could not move temporary output to %s", output_path))
  }
}

process_organ <- function(
    organ_name,
    manifest,
    project_root,
    output_dir,
    cdf_environment) {
  organ_manifest <- manifest[
    tolower(organ) == tolower(organ_name) & include_in_rma
  ]
  if (nrow(organ_manifest) == 0L) {
    stop(sprintf("No RMA samples found for organ %s", organ_name))
  }
  if (anyDuplicated(organ_manifest$sample_id)) {
    stop(sprintf("Duplicate Sample IDs found for organ %s", organ_name))
  }

  cel_files <- resolve_cel_paths(organ_manifest$cel_path, project_root)
  sample_ids <- organ_manifest$sample_id
  cat(sprintf("[%s] RMA samples: %d\n", toupper(organ_name), length(cel_files)))

  assign("rat2302rnensgcdf", cdf_environment, envir = .GlobalEnv)
  expression_set <- justRMA(
    filenames = cel_files,
    celfile.path = "",
    sampleNames = sample_ids,
    cdfname = "rat2302rnensgcdf",
    destructive = TRUE,
    background = TRUE,
    normalize = TRUE,
    bgversion = 2,
    verbose = TRUE
  )
  expression_matrix <- exprs(expression_set)
  if (!identical(colnames(expression_matrix), sample_ids)) {
    stop(sprintf("RMA Sample ID order changed for organ %s", organ_name))
  }

  gene_mask <- !grepl("^AFFX", rownames(expression_matrix), ignore.case = TRUE)
  expression_matrix <- expression_matrix[gene_mask, , drop = FALSE]
  rownames(expression_matrix) <- sub("_at$", "", rownames(expression_matrix))
  if (anyDuplicated(rownames(expression_matrix))) {
    stop(sprintf("Duplicate gene IDs after Brainarray cleanup for organ %s", organ_name))
  }
  if (!all(is.finite(expression_matrix))) {
    stop(sprintf("Non-finite RMA values produced for organ %s", organ_name))
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  output_path <- file.path(output_dir, sprintf("%s_rma_log2.tsv.gz", tolower(organ_name)))
  write_expression_matrix(expression_matrix, output_path)
  cat(sprintf(
    "[%s] Output: %s (%d genes x %d samples)\n",
    toupper(organ_name),
    output_path,
    nrow(expression_matrix),
    ncol(expression_matrix)
  ))

  rm(expression_set, expression_matrix)
  invisible(gc())
}

arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))
project_root <- if (is.null(arguments$project_root)) {
  detect_project_root()
} else {
  normalizePath(arguments$project_root, mustWork = TRUE)
}
manifest_path <- if (is.null(arguments$manifest)) {
  file.path(project_root, "data/expression/metadata/sample_id_manifest.tsv")
} else {
  arguments$manifest
}
output_dir <- if (is.null(arguments$output_dir)) {
  file.path(project_root, "data/expression/rma")
} else {
  arguments$output_dir
}

organ_argument <- tolower(arguments$organ)
if (!(organ_argument %in% c("liver", "kidney", "all"))) {
  stop("--organ must be liver, kidney, or all")
}
organs <- if (organ_argument == "all") c("liver", "kidney") else organ_argument

manifest <- fread(manifest_path, sep = "\t", header = TRUE, colClasses = "character")
required_columns <- c("sample_id", "cel_path", "organ", "include_in_rma")
missing_columns <- setdiff(required_columns, names(manifest))
if (length(missing_columns) > 0L) {
  stop(sprintf("Manifest is missing columns: %s", paste(missing_columns, collapse = ", ")))
}
manifest[, include_in_rma := parse_manifest_boolean(include_in_rma, "include_in_rma")]

cdf_directory <- file.path(project_root, "data/brainarray/Rat2302_Rn_ENSG_25.0.0")
cdf_file <- file.path(cdf_directory, "Rat2302_Rn_ENSG.cdf")
if (!file.exists(cdf_file)) {
  stop(sprintf("Brainarray CDF not found: %s", cdf_file))
}
cat(sprintf("Loading Brainarray CDF: %s\n", cdf_file))
cdf_environment <- make.cdf.env(basename(cdf_file), cdf.path = cdf_directory)

for (organ_name in organs) {
  process_organ(
    organ_name = organ_name,
    manifest = manifest,
    project_root = project_root,
    output_dir = output_dir,
    cdf_environment = cdf_environment
  )
}

cat("RMA processing complete.\n")
