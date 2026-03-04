# utils.R (minimal paper-purpose utilities for GVCM package)
#' Internal utilities for gvcm
#'
#' These functions are not part of the public API.
#'
#' @noRd
NULL

#' @noRd
.as_vec <- function(v, data, n) {
  if (is.character(v) && length(v) == 1L) return(as.numeric(data[[v]]))
  if (is.numeric(v) && length(v) == n) return(as.numeric(v))
  stop("Argument must be a column name (character) or numeric vector of length n.")
}

#' @noRd
.as_Z <- function(Z, data, n) {
  if (is.character(Z)) {
    Zm <- data[, Z, drop = FALSE]
    return(as.matrix(Zm))
  }
  if (is.data.frame(Z)) return(as.matrix(Z))
  if (is.matrix(Z)) return(Z)
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
.make_basis <- function(Zmat, basis = c("ns","poly","none"), df = 5) {
  basis <- match.arg(basis, c("ns","poly","none"))
  p <- ncol(Zmat)

  if (basis == "none") {
    B <- Zmat
    colnames(B) <- paste0("Z", seq_len(p))
    return(B)
  }

  if (!requireNamespace("splines", quietly = TRUE)) {
    stop("Package 'splines' is required for basis expansion.")
  }

  out_list <- vector("list", p)
  for (j in seq_len(p)) {
    zj <- Zmat[, j]
    if (basis == "poly") {
      Bj <- stats::poly(zj, degree = df, raw = FALSE, simple = TRUE)
      Bj <- as.matrix(Bj)
      colnames(Bj) <- paste0("Z", j, "_poly", seq_len(ncol(Bj)))
      out_list[[j]] <- Bj
    } else { # ns
      Bj <- splines::ns(zj, df = df)
      Bj <- as.matrix(Bj)
      colnames(Bj) <- paste0("Z", j, "_ns", seq_len(ncol(Bj)))
      out_list[[j]] <- Bj
    }
  }
  do.call(cbind, out_list)
}

#' @noRd
.make_folds <- function(n, K = 5, seed = 1, strata = NULL) {
  if (K < 2 || K > n) stop("K must be between 2 and n.")
  set.seed(seed)
  if (is.null(strata)) {
    return(sample(rep(seq_len(K), length.out = n)))
  }

  strata <- as.factor(strata)
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
.cv_glmnet_fit <- function(
    x, y,
    family = c("gaussian","binomial","poisson"),
    alpha = 1,
    nfolds = 5,
    lambda_rule = c("lambda.min","lambda.1se")
) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required.")
  }
  family <- match.arg(family, c("gaussian","binomial","poisson"))
  lambda_rule <- match.arg(lambda_rule, c("lambda.min","lambda.1se"))

  fit <- glmnet::cv.glmnet(
    x = x,
    y = y,
    family = family,
    alpha = alpha,
    nfolds = nfolds,
    intercept = TRUE,
    standardize = FALSE
  )
  lam <- if (lambda_rule == "lambda.min") fit$lambda.min else fit$lambda.1se
  list(fit = fit, lambda = lam)
}
