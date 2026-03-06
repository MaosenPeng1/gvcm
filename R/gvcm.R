#' @include utils.R
NULL

#' Generalized Varying Coefficient Model (GVCM) DML estimator (minimal paper version)
#'
#' Fits a generalized varying coefficient model under a canonical GLM link:
#' \eqn{g(\mathbb{E}[Y \mid X, Z]) = \beta_0(Z) + \beta_1(Z)\,X}
#' and estimates the target parameter
#' \eqn{\theta = \mathbb{E}[\beta_1(Z)]}
#' using cross-fitted Double Machine Learning (DML).
#'
#' @param data data.frame
#' @param Y,X outcome/treatment: column name (character) or numeric vector
#' @param Z covariates: column names (character vector) or matrix/data.frame
#' @param link one of c("gaussian","binomial","poisson")
#' @param K number of cross-fitting folds
#' @param basis basis expansion for Z: c("ns","poly","none")
#' @param df Degrees of freedom for the spline basis when \code{basis = "ns"}.
#'   Ignored when \code{basis = "none"} or \code{"poly"}.
#'
#'   If \code{NULL} (default), a link-specific default is used:
#'   \itemize{
#'     \item \code{5} for \code{"gaussian"} and \code{"binomial"} links,
#'     \item \code{3} for the \code{"poisson"} link to reduce instability from the exponential link.
#'   }
#'
#'   Users may specify a larger or smaller value to control the flexibility
#'   of the varying-coefficient functions.
#' @param standardize_Z logical; center/scale raw Z before basis expansion
#' @param alpha Elastic-net mixing parameter used in the \pkg{glmnet} nuisance regressions.
#'   Must be between 0 and 1, where
#'   \itemize{
#'     \item \code{alpha = 0} corresponds to ridge regression,
#'     \item \code{alpha = 1} corresponds to lasso,
#'     \item intermediate values correspond to elastic net.
#'   }
#'   If \code{NULL} (default), a link-specific default is used:
#'   \itemize{
#'     \item \code{0.5} for \code{"gaussian"} and \code{"binomial"} links,
#'     \item \code{0} (ridge) for the \code{"poisson"} link to improve numerical stability.
#'   }
#'   Users may override this by providing an explicit value.
#' @param nfolds_glmnet internal folds for cv.glmnet
#' @param lambda_rule c("lambda.min","lambda.1se")
#' @param crossfit_seed seed for cross-fitting folds
#' @param eps_J lower truncation for \eqn{\hat J(Z)} to avoid division by zero
#' @param verbose logical
#'
#' @return A list with \code{theta_hat}, \code{se_hat}, and \code{ci}.
#' @export
gvcm <- function(
    data,
    Y, X, Z,
    link = c("gaussian","binomial","poisson"),
    K = 5,
    basis = c("ns","poly","none"),
    df = NULL,
    standardize_Z = TRUE,
    alpha = NULL,
    nfolds_glmnet = 5,
    lambda_rule = c("lambda.min","lambda.1se"),
    crossfit_seed = 1,
    eps_J = 1e-8,
    verbose = TRUE
) {
  # -----------------------
  # Dependencies
  # -----------------------
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("Package 'glmnet' is required.")
  }
  if (!requireNamespace("splines", quietly = TRUE)) {
    stop("Package 'splines' is required (for basis='ns').")
  }

  # -----------------------
  # Local helpers (model-specific only)
  # -----------------------
  .fit_beta_model <- function(B_tr, X_tr, Y_tr, link, alpha, nfolds, lambda_rule) {
    family_y <- match.arg(link, c("gaussian","binomial","poisson"))

    XB_tr <- B_tr * as.numeric(X_tr)
    D_tr <- cbind(B_tr, XB_tr)
    colnames(D_tr) <- c(
      paste0(colnames(B_tr), "_b0"),
      paste0(colnames(B_tr), "_b1")
    )

    cv <- .cv_glmnet_fit(
      x = D_tr, y = Y_tr,
      family = family_y,
      alpha = alpha, nfolds = nfolds, lambda_rule = lambda_rule
    )
    list(cv = cv, pB = ncol(B_tr))
  }

  .predict_eta_beta1 <- function(beta_obj, B_te, X_te) {
    pB <- beta_obj$pB

    XB_te <- B_te * as.numeric(X_te)
    D_te <- cbind(B_te, XB_te)

    eta_hat <- as.numeric(stats::predict(
      beta_obj$cv$fit,
      newx = D_te,
      s = beta_obj$cv$lambda,
      type = "link"
    ))

    coefs <- as.matrix(glmnet::coef.glmnet(
      beta_obj$cv$fit$glmnet.fit,
      s = beta_obj$cv$lambda
    ))
    # Rows: 1 intercept; next pB for beta0; next pB for beta1
    beta1_coef <- coefs[(1 + pB + 1):(1 + 2 * pB), 1, drop = TRUE]
    beta1_hat <- as.numeric(B_te %*% beta1_coef)

    list(eta_hat = eta_hat, beta1_hat = beta1_hat)
  }

  .fit_m_model <- function(B_tr, X_tr, alpha, nfolds, lambda_rule) {
    .cv_glmnet_fit(
      x = B_tr, y = as.numeric(X_tr),
      family = "gaussian",
      alpha = alpha, nfolds = nfolds, lambda_rule = lambda_rule
    )
  }

  .predict_m <- function(m_obj, B_te) {
    as.numeric(stats::predict(m_obj$fit, newx = B_te, s = m_obj$lambda, type = "response"))
  }

  .fit_invar_model <- function(B_tr, Y_tr, weights, alpha, nfolds, lambda_rule) {
    .cv_glmnet_fit(
      x = B_tr,
      y = Y_tr,
      weights = weights,
      family = "gaussian",
      alpha = alpha,
      nfolds = nfolds,
      lambda_rule = lambda_rule
    )
  }

  .predict_invar <- function(invar_obj, B_te) {
    as.numeric(stats::predict(invar_obj$fit, newx = B_te, s = invar_obj$lambda, type = "response"))
  }

  # -----------------------
  # 1) Parse inputs + drop missing
  # -----------------------
  link <- match.arg(link, c("gaussian","binomial","poisson"))
  basis <- match.arg(basis, c("ns","poly","none"))
  lambda_rule <- match.arg(lambda_rule, c("lambda.min","lambda.1se"))

  # -----------------------
  # Link-specific defaults (only if not provided)
  # -----------------------
  if (is.null(alpha)) {
    alpha <- if (link == "poisson") 0 else 0.5
  }

  if (is.null(df)) {
    if (basis == "ns") {
      df <- if (link == "poisson") 3 else 5
    } else {
      df <- 5  # df unused for basis="none"; safe placeholder
    }
  }

  if (!is.data.frame(data)) stop("data must be a data.frame.")
  n0 <- nrow(data)

  Yv <- .as_vec(Y, data, n0)
  Xv <- .as_vec(X, data, n0)
  Zv <- .as_Z(Z, data, n0)

  keep <- stats::complete.cases(cbind(Yv, Xv, Zv))
  Yv <- Yv[keep]
  Xv <- Xv[keep]
  Zv <- Zv[keep, , drop = FALSE]
  n <- length(Yv)

  if (n < 10) stop("Too few complete cases after dropping missingness.")
  if (K < 2 || K > n) stop("K must be between 2 and n.")

  # -----------------------
  # 2) Build basis
  # -----------------------
  if (standardize_Z) {
    st <- .standardize(Zv)
    Zv_use <- st$M
  } else {
    Zv_use <- Zv
  }

  B <- .make_basis(Zv_use, basis = basis, df = df)
  B <- as.matrix(B)

  # -----------------------
  # 3) Create folds
  # -----------------------
  if (link == "poisson") {
    fold_id <- .make_folds(n, K = K, seed = crossfit_seed, strata = (Yv > 0))
  } else {
    fold_id <- .make_folds(n, K = K, seed = crossfit_seed)
  }

  # Storage for cross-fitted predictions
  beta1_hat <- rep(NA_real_, n)
  mu_hat    <- rep(NA_real_, n)
  m_hat     <- rep(NA_real_, n)
  J_te_inv  <- rep(NA_real_, n)

  if (verbose) message("gvcm: cross-fitting with K = ", K, " folds; link = ", link, "; basis = ", basis)

  # -----------------------
  # 4) Cross-fitting loop
  # -----------------------
  for (k in seq_len(K)) {
    idx_te <- which(fold_id == k)
    idx_tr <- which(fold_id != k)

    B_tr <- B[idx_tr, , drop = FALSE]
    B_te <- B[idx_te, , drop = FALSE]
    X_tr <- Xv[idx_tr]
    X_te <- Xv[idx_te]
    Y_tr <- Yv[idx_tr]

    # (a) Fit beta model on training
    beta_obj <- .fit_beta_model(
      B_tr = B_tr, X_tr = X_tr, Y_tr = Y_tr,
      link = link,
      alpha = alpha, nfolds = nfolds_glmnet, lambda_rule = lambda_rule
    )

    # Predict eta on train (for pseudo outcome) and eta/beta1 on test
    pred_tr <- .predict_eta_beta1(beta_obj, B_te = B_tr, X_te = X_tr)
    pred_te <- .predict_eta_beta1(beta_obj, B_te = B_te, X_te = X_te)

    mu_tr <- .inv_link(pred_tr$eta_hat, link = link)
    mu_te <- .inv_link(pred_te$eta_hat, link = link)

    # (b) Fit m(Z)=E[X|Z] on training
    m_obj <- .fit_m_model(
      B_tr = B_tr, X_tr = X_tr,
      alpha = alpha, nfolds = nfolds_glmnet, lambda_rule = lambda_rule
    )
    m_tr <- .predict_m(m_obj, B_tr)
    m_te <- .predict_m(m_obj, B_te)

    # (c) Build pseudo outcome for J on training: T = (X - mhat)^2 * V(muhat)
    V_te <- .V_fun(mu_te, link = link)
    r <- X_tr - m_tr

    # (d) Fit J(Z)=E[T|Z] on training, predict on test
    Y_tr_inv <- 1/(r^2+eps_J)
    wt <- r^2+eps_J
    invar_obj <- .fit_invar_model(
      B_tr = B_tr, Y_tr = Y_tr_inv,
      alpha = alpha, nfolds = nfolds_glmnet, lambda_rule = lambda_rule, weights = wt
    )
    invar_te <- .predict_invar(invar_obj, B_te)
    J_inv_te_fold <- invar_te / V_te
    J_inv_te_fold <- pmax(J_inv_te_fold, eps_J)

    # Store test-fold predictions
    beta1_hat[idx_te] <- pred_te$beta1_hat
    mu_hat[idx_te]    <- mu_te
    m_hat[idx_te]     <- m_te
    J_te_inv[idx_te]  <- J_inv_te_fold

  }

  if (any(!is.finite(beta1_hat)) || any(!is.finite(mu_hat)) ||
      any(!is.finite(m_hat))     || any(!is.finite(J_te_inv))) {
    stop("Non-finite nuisance predictions encountered (check eps_J, basis, and glmnet fits).")
  }

  # -----------------------
  # 5) Compute theta + se via EIF
  # EIF (canonical): phi_i = beta1(Z_i) - theta + (X - m(Z)) * (Y - mu) / J(Z)
  # theta_hat = mean( beta1_hat + (X - m_hat)*(Y - mu_hat)/J_hat )
  # -----------------------
  X_tilde <- Xv - m_hat
  adj_term <- X_tilde * (Yv - mu_hat) * J_te_inv

  theta_hat <- mean(beta1_hat + adj_term)

  IF <- (beta1_hat - theta_hat) + adj_term
  se_hat <- stats::sd(IF) / sqrt(n)

  ci <- c(theta_hat - 1.96 * se_hat, theta_hat + 1.96 * se_hat)

  # -----------------------
  # 6) Return
  # -----------------------
  list(
    theta_hat = theta_hat,
    se_hat    = se_hat,
    ci        = ci,
    n         = n,
    K         = K,
    link      = link
  )
}
