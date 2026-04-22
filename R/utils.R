# utils.R (minimal paper-purpose utilities for GVCM package)
#' Internal utilities for gvcm
#'
#' These functions are not part of the public API.
#'
#' @noRd
NULL

#' @noRd
.as_vec <- function(v, data, n) {
  if (is.character(v) && length(v) == 1L) {
    if (!v %in% names(data)) stop("Column '", v, "' not found in data.")
    val <- data[[v]]
    if (is.factor(val)) stop("Column '", v, "' must be numeric, not factor.")
    if (!is.numeric(val)) stop("Column '", v, "' must be numeric.")
    return(as.numeric(val))
  }
  if (is.numeric(v) && length(v) == n) return(as.numeric(v))
  stop("Argument must be a column name (character) or numeric vector of length n.")
}

#' @noRd
.as_Z <- function(Z, data, n) {
  if (is.character(Z)) {
    if (!all(Z %in% names(data))) {
      stop("Some Z columns not found in data.")
    }
    Zm <- data[, Z, drop = FALSE]
    if (!all(vapply(Zm, is.numeric, logical(1)))) {
      stop("All Z columns must be numeric.")
    }
    if (nrow(Zm) != n) stop("Z must have n rows.")
    return(as.matrix(Zm))
  }
  if (is.data.frame(Z)) {
    if (!all(vapply(Z, is.numeric, logical(1)))) {
      stop("All columns of Z data.frame must be numeric.")
    }
    if (nrow(Z) != n) stop("Z must have n rows.")
    return(as.matrix(Z))
  }
  if (is.matrix(Z)) {
    if (!is.numeric(Z)) stop("Z matrix must be numeric.")
    if (nrow(Z) != n) stop("Z must have n rows.")
    return(Z)
  }
  stop("Z must be column names, a data.frame, or a matrix.")
}

#' @noRd
.standardize <- function(M) {
  mu <- colMeans(M)
  sdv <- apply(M, 2L, stats::sd)
  sdv[sdv == 0] <- 1
  list(
    M = sweep(sweep(M, 2L, mu, "-"), 2L, sdv, "/"),
    center = mu,
    scale = sdv
  )
}

#' @noRd
.make_basis <- function(Zmat, basis = "ns", df = 4) {
  if (!is.matrix(Zmat)) Zmat <- as.matrix(Zmat)
  if (!is.numeric(Zmat)) stop("Z matrix must be numeric.")
  n <- nrow(Zmat)
  p <- ncol(Zmat)

  # Give Z columns stable names if missing
  if (is.null(colnames(Zmat))) {
    colnames(Zmat) <- paste0("Z", seq_len(p))
  }

  # -------------------------
  # Case 1: basis is a built-in character option
  # -------------------------
  if (is.character(basis) && length(basis) == 1L) {
    basis <- match.arg(basis, c("ns","poly","none"))

    if (basis == "none") {
      B <- Zmat
      return(B)
    }

    if (basis == "poly") {
      if (!is.numeric(df) || length(df) != 1L || df < 1) {
        stop("df must be a single positive integer for poly basis.")
      }
      out_list <- vector("list", p)
      for (j in seq_len(p)) {
        zj <- Zmat[, j]
        Bj <- tryCatch(
          stats::poly(zj, degree = df, raw = FALSE, simple = TRUE),
          error = function(e) stop(
            "poly basis construction failed for ", colnames(Zmat)[j], ": ", e$message,
            call. = FALSE
          )
        )
        Bj <- as.matrix(Bj)
        colnames(Bj) <- paste0(colnames(Zmat)[j], "_poly", seq_len(ncol(Bj)))
        out_list[[j]] <- Bj
      }
      return(do.call(cbind, out_list))
    }

    if (basis == "ns") {
      if (!is.numeric(df) || length(df) != 1L || df < 1) {
        stop("df must be a single positive integer for ns basis.")
      }
      if (!requireNamespace("splines", quietly = TRUE)) {
        stop("Package 'splines' is required for basis = 'ns'.")
      }
      out_list <- vector("list", p)
      for (j in seq_len(p)) {
        zj <- Zmat[, j]
        Bj <- tryCatch(
          splines::ns(zj, df = df),
          error = function(e) stop(
            "ns basis construction failed for ", colnames(Zmat)[j], ": ", e$message,
            call. = FALSE
          )
        )
        Bj <- as.matrix(Bj)
        colnames(Bj) <- paste0(colnames(Zmat)[j], "_ns", seq_len(ncol(Bj)))
        out_list[[j]] <- Bj
      }
      return(do.call(cbind, out_list))
    }
  }

  # -------------------------
  # Case 2: basis is a formula
  # -------------------------
  if (inherits(basis, "formula")) {
    Zdf <- as.data.frame(Zmat)
    mm <- stats::model.matrix(basis, data = Zdf)

    # remove intercept if present
    if ("(Intercept)" %in% colnames(mm)) {
      mm <- mm[, colnames(mm) != "(Intercept)", drop = FALSE]
    }

    if (!is.matrix(mm) || nrow(mm) != n) {
      stop("Custom formula basis must produce a matrix with nrow(Zmat) rows.")
    }
    return(mm)
  }

  # -------------------------
  # Case 3: basis is a function
  # -------------------------
  if (is.function(basis)) {
    B <- basis(Zmat)
    if (is.data.frame(B)) B <- as.matrix(B)
    if (!is.matrix(B) || !is.numeric(B) || nrow(B) != n) {
      stop("Custom basis function must return a numeric matrix with nrow(Zmat) rows.")
    }
    if (is.null(colnames(B))) {
      colnames(B) <- paste0("B", seq_len(ncol(B)))
    }
    return(B)
  }

  # -------------------------
  # Case 4: basis is already a matrix/data.frame
  # -------------------------
  if (is.data.frame(basis)) basis <- as.matrix(basis)
  if (is.matrix(basis)) {
    if (!is.numeric(basis) || nrow(basis) != n) {
      stop("If basis is supplied as a matrix, it must be numeric and have nrow(Zmat) rows.")
    }
    if (is.null(colnames(basis))) {
      colnames(basis) <- paste0("B", seq_len(ncol(basis)))
    }
    return(basis)
  }

  stop(
    "basis must be one of 'ns', 'poly', 'none', or a formula, function, matrix, or data.frame."
  )
}

#' @noRd
.make_folds <- function(n, K = 5, seed = 1, strata = NULL) {
  if (K < 2 || K > n) stop("K must be between 2 and n.")
  set.seed(seed)
  if (is.null(strata)) {
    return(sample(rep(seq_len(K), length.out = n)))
  }

  strata <- as.factor(strata)
  if (length(strata) != n) stop("strata must have length n.")
  tab <- table(strata)
  if (any(tab < K)) {
    warning(
      "Some strata have fewer observations than K; falling back to unstratified folds."
    )
    return(sample(rep(seq_len(K), length.out = n)))
  }

  foldid <- integer(n)
  # assign folds within each stratum to balance composition across folds
  for (lv in levels(strata)) {
    idx <- which(strata == lv)
    foldid[idx] <- sample(rep(seq_len(K), length.out = length(idx)))
  }

  foldid
}

#' @noRd
.inv_link <- function(eta, link = c("gaussian","binomial","poisson")) {
  link <- match.arg(link, c("gaussian","binomial","poisson"))
  if (link == "gaussian") return(eta)
  if (link == "binomial") return(1 / (1 + exp(-eta)))
  pmax(exp(eta), 1e-12)
}

#' @noRd
.V_fun <- function(mu, link = c("gaussian","binomial","poisson")) {
  link <- match.arg(link, c("gaussian","binomial","poisson"))
  if (link == "gaussian") return(rep(1, length(mu)))
  if (link == "binomial") return(pmax(mu * (1 - mu), 1e-12))
  pmax(mu, 1e-12) # poisson
}

#' @noRd
.fill_defaults <- function(defaults, x) {
  if (is.null(x)) x <- list()
  utils::modifyList(defaults, x)
}
