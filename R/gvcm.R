#' @include utils.R
#' @include learner_dispatch.R
#' @include sieve_models.R
#' @include deep_models.R
NULL

#' Generalized Varying Coefficient Model (GVCM) DML estimator
#'
#' Fits the model
#' \deqn{g\{E(Y \mid X, Z)\} = \beta_0(Z) + \beta_1(Z) X}
#' and targets
#' \deqn{\theta = E\{\beta_1(Z)\}.}
#'
#' The estimator uses cross-fitting and an efficient influence function (EIF)
#' with nuisance components estimated by either sieve-based learners or
#' deep neural networks.
#'
#' Supported links are:
#' \itemize{
#'   \item \code{"gaussian"} with identity link,
#'   \item \code{"binomial"} with logit link,
#'   \item \code{"poisson"} with log link.
#' }
#'
#' @param data A \code{data.frame} containing the analysis variables.
#' @param Y Outcome variable, supplied either as a column name (character)
#'   in \code{data} or as a numeric vector of length \code{nrow(data)}.
#' @param X Exposure/treatment variable, supplied either as a column name
#'   (character) in \code{data} or as a numeric vector of length \code{nrow(data)}.
#' @param Z Covariates, supplied as a character vector of column names, or as
#'   a numeric matrix/data.frame with \code{nrow(data)} rows.
#' @param link One of \code{c("gaussian","binomial","poisson")}.
#' @param learner Nuisance learner type. One of \code{c("sieve","deepnet")}.
#' @param K Number of cross-fitting folds.
#' @param standardize_Z Logical; if \code{TRUE}, raw \code{Z} is centered and
#'   scaled before basis expansion (for sieve) or before being passed into the
#'   deep network learner.
#' @param crossfit_seed Random seed used to generate cross-fitting folds.
#' @param eps_J Small positive stabilization constant used when estimating and
#'   lower-truncating the inverse canonical information \eqn{1/D(Z)}. The name
#'   is retained for backward compatibility.
#' @param sieve_args A named list of tuning parameters for the sieve learner.
#'   Supported components are:
#'   \itemize{
#'     \item \code{basis}: either one of \code{"ns"}, \code{"poly"}, \code{"none"},
#'       or a custom basis supplied as a formula, function, matrix, or data.frame,
#'     \item \code{df}: degrees of freedom for \code{"ns"} or \code{"poly"},
#'     \item \code{alpha}: elastic-net mixing parameter for \pkg{glmnet},
#'     \item \code{nfolds_glmnet}: internal folds for \code{cv.glmnet},
#'     \item \code{lambda_rule}: one of \code{"lambda.min"} or \code{"lambda.1se"}.
#'   }
#' @param net_args A named list of tuning parameters for the deep neural network
#'   learner. Supported components are:
#'   \itemize{
#'     \item \code{hidden_dims}: integer vector giving hidden-layer widths,
#'     \item \code{dropout}: dropout rate in \code{[0,1)},
#'     \item \code{n_residual}: number of residual blocks applied after the shared trunk,
#'     \item \code{lr}: learning rate for Adam,
#'     \item \code{epochs}: maximum number of training epochs,
#'     \item \code{batch_size}: mini-batch size,
#'     \item \code{weight_decay}: weight decay passed to Adam,
#'     \item \code{valid_prop}: proportion of the training fold used as a validation split,
#'     \item \code{early_stop_patience}: early stopping patience,
#'     \item \code{min_delta}: minimum validation improvement required to reset patience,
#'     \item \code{device}: computation device, typically \code{"cpu"} or \code{"cuda"},
#'     \item \code{seed}: random seed used inside network fitting,
#'     \item \code{verbose}: logical; if \code{TRUE}, training progress is printed.
#'   }
#' @param beta_net_args Optional named list of deepnet tuning overrides used
#'   only for the beta-network nuisance learner. These are merged on top of
#'   \code{net_args}. Useful components include:
#'   \itemize{
#'     \item \code{activation}: e.g. \code{"silu"}, \code{"relu"}, \code{"gelu"}, \code{"tanh"},
#'     \item \code{lambda_beta1}: nonnegative penalty weight applied to the
#'       slope head \eqn{\beta_1(Z)} during beta-network training,
#'     \item any other components supported in \code{net_args}.
#'   }
#' @param invar_net_args Optional named list of deepnet tuning overrides used
#'   only for the inverse-information nuisance learner. These are merged on top
#'   of \code{net_args}.
#' @param r_net_args Optional named list of deepnet tuning overrides used only
#'   for the information-weighted exposure nuisance
#'   \deqn{r(Z)=\frac{E\{X V(\mu)\mid Z\}}{E\{V(\mu)\mid Z\}}.}
#'   These are merged on top of \code{net_args}. For the Gaussian identity link,
#'   this reduces to \eqn{E(X\mid Z)}.
#'
#' @return A list with components
#' \itemize{
#'   \item \code{theta_hat}: estimated target parameter,
#'   \item \code{se_hat}: estimated standard error,
#'   \item \code{ci}: Wald-type 95\% confidence interval,
#'   \item \code{n}: number of complete cases used,
#'   \item \code{K}: number of cross-fitting folds,
#'   \item \code{link}: link used,
#'   \item \code{learner}: nuisance learner used,
#'   \item \code{beta1_hat}: cross-fitted estimates of \eqn{\beta_1(Z_i)},
#'   \item \code{r_hat}: cross-fitted estimates of the information-weighted
#'     exposure regression \eqn{r(Z_i)},
#'   \item \code{mu_hat}: cross-fitted estimates of \eqn{E(Y\mid X_i,Z_i)},
#'   \item \code{D_inv_hat}: cross-fitted estimates of \eqn{1/D(Z_i)},\code{r_hat}
#'    \code{J_inv_hat}: deprecated aliases for \code{D_inv_hat}.
#' }
#' @export
gvcm <- function(
    data,
    Y, X, Z,
    link = c("gaussian","binomial","poisson"),
    learner = c("sieve","deepnet"),
    K = 5,
    standardize_Z = TRUE,
    crossfit_seed = 1,
    eps_J = 1e-6,
    sieve_args = list(
      basis = "ns",
      df = NULL,
      alpha = 1,
      nfolds_glmnet = 5,
      lambda_rule = "lambda.min"
    ),
    net_args = list(
      hidden_dims = c(64,32),
      dropout = 0,
      n_residual = 1,
      activation = "tanh",
      lr = 5e-4,
      epochs = 400,
      batch_size = 64,
      weight_decay = 1e-4,
      valid_prop = 0.15,
      early_stop_patience = 20,
      min_delta = 1e-4,
      device = "cpu",
      seed = 1,
      verbose = FALSE
    ),
    beta_net_args = list(
      lambda_beta1 = 1e-4
    ),
    invar_net_args = list(),
    r_net_args = NULL
)
{
  # -----------------------
  # Dependencies
  # -----------------------
  learner <- match.arg(learner, c("sieve", "deepnet"))
  link    <- match.arg(link, c("gaussian","binomial","poisson"))

  if (!is.data.frame(data)) stop("data must be a data.frame.")

  if (learner == "sieve") {
    if (!requireNamespace("glmnet", quietly = TRUE)) {
      stop("Package 'glmnet' is required for learner = 'sieve'.")
    }
  }

  if (learner == "deepnet") {
    if (!requireNamespace("torch", quietly = TRUE)) {
      stop("Package 'torch' is required for learner = 'deepnet'.")
    }
  }

  # -----------------------
  # 0) Resolve learner-specific arguments
  # -----------------------
  sieve_defaults <- list(
    basis = "ns",
    df = NULL,
    alpha = 1,
    nfolds_glmnet = 5,
    lambda_rule = "lambda.min"
  )

  net_defaults <- list(
    hidden_dims = c(64, 32),
    dropout = 0,
    n_residual = 1,
    activation = "tanh",
    lr = 5e-4,
    epochs = 400,
    batch_size = 32,
    weight_decay = 1e-4,
    valid_prop = 0.15,
    early_stop_patience = 15,
    min_delta = 1e-4,
    device = "cpu",
    seed = 1,
    verbose = FALSE
  )

  sieve_args <- .fill_defaults(sieve_defaults, sieve_args)
  net_args   <- .fill_defaults(net_defaults, net_args)

  if (is.null(r_net_args)) r_net_args <- list()

  if (!is.numeric(eps_J) || length(eps_J) != 1L || eps_J <= 0) {
    stop("eps_J must be a single positive number.")
  }
  if (learner == "deepnet") {
    beta_net_args  <- .fill_defaults(net_args, beta_net_args)
    r_net_args     <- .fill_defaults(net_args, r_net_args)
    invar_net_args <- .fill_defaults(net_args, invar_net_args)

    # beta-only default if not supplied by user/global args
    if (is.null(beta_net_args$lambda_beta1)) {
      beta_net_args$lambda_beta1 <- 1e-4
    }
  }

  # -----------------------
  # 0a) Validate sieve args
  # -----------------------
  if (learner == "sieve") {
    # allow basis to be:
    # "ns", "poly", "none", or a formula/function/matrix/data.frame
    if (is.character(sieve_args$basis) && length(sieve_args$basis) == 1L) {
      sieve_args$basis <- match.arg(sieve_args$basis, c("ns","poly","none"))
    } else if (
      !inherits(sieve_args$basis, "formula") &&
      !is.function(sieve_args$basis) &&
      !is.matrix(sieve_args$basis) &&
      !is.data.frame(sieve_args$basis)
    ) {
      stop(
        "sieve_args$basis must be one of 'ns', 'poly', 'none', ",
        "or a formula, function, matrix, or data.frame."
      )
    }

    sieve_args$lambda_rule <- match.arg(
      sieve_args$lambda_rule,
      c("lambda.min","lambda.1se")
    )

    if (is.null(sieve_args$df)) {
      sieve_args$df <- 4
    }

    if (!is.numeric(sieve_args$alpha) || length(sieve_args$alpha) != 1L ||
        sieve_args$alpha < 0 || sieve_args$alpha > 1) {
      stop("sieve_args$alpha must be a single number in [0, 1].")
    }

    if (!is.numeric(sieve_args$nfolds_glmnet) ||
        length(sieve_args$nfolds_glmnet) != 1L ||
        sieve_args$nfolds_glmnet < 2) {
      stop("sieve_args$nfolds_glmnet must be a single integer >= 2.")
    }

    if (is.character(sieve_args$basis) && identical(sieve_args$basis, "ns")) {
      if (!requireNamespace("splines", quietly = TRUE)) {
        stop("Package 'splines' is required for sieve_args$basis = 'ns'.")
      }
    }
  }

  # convenient local aliases: sieve
  if (learner == "sieve") {
    basis         <- sieve_args$basis
    df            <- sieve_args$df
    alpha         <- sieve_args$alpha
    nfolds_glmnet <- sieve_args$nfolds_glmnet
    lambda_rule   <- sieve_args$lambda_rule
  }

  # -----------------------
  # 0b) Validate deepnet args
  # -----------------------
  if (learner == "deepnet") {
    if (!is.numeric(net_args$hidden_dims) || length(net_args$hidden_dims) < 1L) {
      stop("net_args$hidden_dims must be a numeric vector of positive integers.")
    }
    if (any(net_args$hidden_dims <= 0)) {
      stop("All entries of net_args$hidden_dims must be positive.")
    }

    if (!is.numeric(net_args$dropout) || length(net_args$dropout) != 1L ||
        net_args$dropout < 0 || net_args$dropout >= 1) {
      stop("net_args$dropout must be a single number in [0, 1).")
    }

    if (!is.numeric(net_args$lr) || length(net_args$lr) != 1L ||
        net_args$lr <= 0) {
      stop("net_args$lr must be a positive number.")
    }

    if (!is.numeric(net_args$epochs) || length(net_args$epochs) != 1L ||
        net_args$epochs < 1) {
      stop("net_args$epochs must be a positive integer.")
    }

    if (!is.numeric(net_args$batch_size) || length(net_args$batch_size) != 1L ||
        net_args$batch_size < 1) {
      stop("net_args$batch_size must be a positive integer.")
    }

    if (!is.numeric(net_args$weight_decay) ||
        length(net_args$weight_decay) != 1L ||
        net_args$weight_decay < 0) {
      stop("net_args$weight_decay must be a nonnegative number.")
    }
    if (!is.numeric(net_args$n_residual) || length(net_args$n_residual) != 1L ||
        net_args$n_residual < 0) {
      stop("net_args$n_residual must be a single nonnegative integer.")
    }
    if (!is.character(net_args$activation) || length(net_args$activation) != 1L) {
      stop("net_args$activation must be a single character string.")
    }
    net_args$activation <- match.arg(
      net_args$activation,
      c("silu", "relu", "gelu", "tanh")
    )
    beta_net_args$activation <- match.arg(
      beta_net_args$activation,
      c("silu", "relu", "gelu", "tanh")
    )
    r_net_args$activation <- match.arg(
      r_net_args$activation,
      c("silu", "relu", "gelu", "tanh")
    )
    invar_net_args$activation <- match.arg(
      invar_net_args$activation,
      c("silu", "relu", "gelu", "tanh")
    )

    if (!is.numeric(beta_net_args$lambda_beta1) ||
        length(beta_net_args$lambda_beta1) != 1L ||
        beta_net_args$lambda_beta1 < 0) {
      stop("beta_net_args$lambda_beta1 must be a single nonnegative number.")
    }

    if (!is.numeric(net_args$valid_prop) || length(net_args$valid_prop) != 1L ||
        net_args$valid_prop <= 0 || net_args$valid_prop >= 1) {
      stop("net_args$valid_prop must be a single number in (0, 1).")
    }

    if (!is.numeric(net_args$early_stop_patience) ||
        length(net_args$early_stop_patience) != 1L ||
        net_args$early_stop_patience < 1) {
      stop("net_args$early_stop_patience must be a positive integer.")
    }

    if (!is.numeric(net_args$min_delta) || length(net_args$min_delta) != 1L ||
        net_args$min_delta < 0) {
      stop("net_args$min_delta must be a nonnegative number.")
    }

    if (!is.character(net_args$device) || length(net_args$device) != 1L) {
      stop("net_args$device must be a single character string.")
    }

    if (!is.numeric(net_args$seed) || length(net_args$seed) != 1L) {
      stop("net_args$seed must be a single numeric value.")
    }

    if (!is.logical(net_args$verbose) || length(net_args$verbose) != 1L) {
      stop("net_args$verbose must be TRUE or FALSE.")
    }
  }

  # -----------------------
  # 1) Parse inputs + drop missing
  # -----------------------
  n0 <- nrow(data)

  Yv <- .as_vec(Y, data, n0)
  Xv <- .as_vec(X, data, n0)
  Zv <- .as_Z(Z, data, n0)

  keep <- stats::complete.cases(cbind(Yv, Xv, Zv))
  finite_keep <- is.finite(Yv) & is.finite(Xv) & rowSums(!is.finite(Zv)) == 0
  keep <- keep & finite_keep
  Yv <- Yv[keep]
  Xv <- Xv[keep]
  Zv <- Zv[keep, , drop = FALSE]
  n <- length(Yv)

  if (n < 10) stop("Too few complete cases after dropping missingness.")
  if (K < 2 || K > n) stop("K must be between 2 and n.")

  # -----------------------
  # 2) Preprocess Z
  # -----------------------
  if (standardize_Z) {
    st <- .standardize(Zv)
    Zv_use <- st$M
  } else {
    st <- NULL
    Zv_use <- Zv
  }

  # basis matrix is only needed for sieve learner
  if (learner == "sieve") {
    B <- .make_basis(Zv_use, basis = basis, df = df)
    B <- as.matrix(B)
  } else {
    B <- NULL
  }

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
  r_hat     <- rep(NA_real_, n)
  D_inv_hat <- rep(NA_real_, n)

  # -----------------------
  # 4) Cross-fitting loop
  # -----------------------
  for (k in seq_len(K)) {
    idx_te <- which(fold_id == k)
    idx_tr <- which(fold_id != k)

    # learner-specific feature inputs
    if (learner == "sieve") {
      B_tr <- B[idx_tr, , drop = FALSE]
      B_te <- B[idx_te, , drop = FALSE]
    } else {
      B_tr <- NULL
      B_te <- NULL
    }

    Z_tr <- Zv_use[idx_tr, , drop = FALSE]
    Z_te <- Zv_use[idx_te, , drop = FALSE]

    X_tr <- Xv[idx_tr]
    X_te <- Xv[idx_te]
    Y_tr <- Yv[idx_tr]

    # -----------------------
    # (a) Fit beta model on training
    # -----------------------
    beta_obj <- .fit_beta_model(
      learner    = learner,
      B_tr       = B_tr,
      Z_tr       = Z_tr,
      X_tr       = X_tr,
      Y_tr       = Y_tr,
      link       = link,
      sieve_args = sieve_args,
      net_args   = beta_net_args
    )

    beta_pred_te <- .predict_beta_model(
      fit     = beta_obj,
      learner = learner,
      B_te    = B_te,
      Z_te    = Z_te,
      X_te    = X_te
    )

    # Training-fold fitted means determine the information weights used to
    # estimate r(Z). No held-out outcomes enter this nuisance fit.
    beta_pred_tr <- .predict_beta_model(
      fit     = beta_obj,
      learner = learner,
      B_te    = B_tr,
      Z_te    = Z_tr,
      X_te    = X_tr
    )

    beta1_hat_te <- beta_pred_te$beta1_hat
    mu_hat_tr <- .mu_from_eta(beta_pred_tr$eta_hat, link = link)
    mu_hat_te <- .mu_from_eta(beta_pred_te$eta_hat, link = link)
    r_weights_tr <- .r_weights(mu_hat_tr, link = link)

    # -----------------------
    # (b) Fit the information-weighted r(Z) on training:
    #     r(Z) = E[X V(mu) | Z] / E[V(mu) | Z].
    #     For Gaussian identity only, this equals E[X | Z].
    # -----------------------
    r_obj <- .fit_r_model(
      learner    = learner,
      B_tr       = B_tr,
      Z_tr       = Z_tr,
      X_tr       = X_tr,
      r_weights_tr = r_weights_tr,
      sieve_args = sieve_args,
      net_args   = r_net_args
    )

    r_hat_tr <- .predict_r_model(
      fit     = r_obj,
      learner = learner,
      B_te    = B_tr,
      Z_te    = Z_tr
    )

    r_hat_te <- .predict_r_model(
      fit     = r_obj,
      learner = learner,
      B_te    = B_te,
      Z_te    = Z_te
    )

    # -----------------------
    # (c) Build the inverse-D regression target on training, where
    #     D(Z) = E[(X - r(Z))^2 V(mu) | Z].
    #
    # Weighted least squares of 1 / A on Z with weights A targets
    # 1 / E[A | Z], here with A = (X - r_hat)^2 V(mu_hat) + eps_J.
    # -----------------------
    x_minus_r_tr <- X_tr - r_hat_tr
    D_contrib_tr <- x_minus_r_tr^2 * r_weights_tr
    D_stable_tr  <- D_contrib_tr + eps_J
    Y_tr_inv     <- 1 / D_stable_tr
    wt_tr        <- D_stable_tr

    # -----------------------
    # (d) Fit D^{-1}(Z) via weighted regression on training.
    # -----------------------
    invar_obj <- .fit_invar_model(
      learner    = learner,
      B_tr       = B_tr,
      Z_tr       = Z_tr,
      Y_tr       = Y_tr_inv,
      weights    = wt_tr,
      sieve_args = sieve_args,
      net_args   = invar_net_args
    )

    invar_te <- .predict_invar_model(
      fit     = invar_obj,
      learner = learner,
      B_te    = B_te,
      Z_te    = Z_te
    )

    D_inv_te_fold <- pmax(invar_te, eps_J)

    # -----------------------
    # store fold-specific predictions
    # -----------------------
    beta1_hat[idx_te] <- beta1_hat_te
    mu_hat[idx_te]    <- mu_hat_te
    r_hat[idx_te]     <- r_hat_te
    D_inv_hat[idx_te] <- D_inv_te_fold
  }

  if (any(!is.finite(beta1_hat)) || any(!is.finite(mu_hat)) ||
      any(!is.finite(r_hat))     || any(!is.finite(D_inv_hat))) {
    stop("Non-finite nuisance predictions encountered. Check eps_J, learner settings, and nuisance model fits.")
  }

  # -----------------------
  # 5) Compute theta + se via EIF
  # EIF (canonical-link representation):
  #   phi_i = beta1(Z_i) - theta
  #           + (X_i - r(Z_i)) * (Y_i - mu_i) / D(Z_i)
  # Since D_inv_hat stores 1 / D_hat(Z_i), we compute:
  #   theta_hat = mean(
  #     beta1_hat + (X - r_hat) * (Y - mu_hat) * D_inv_hat
  #   )
  # -----------------------
  X_tilde  <- Xv - r_hat
  adj_term <- X_tilde * (Yv - mu_hat) * D_inv_hat

  theta_hat <- mean(beta1_hat + adj_term)

  IF <- (beta1_hat - theta_hat) + adj_term
  se_hat <- stats::sd(IF) / sqrt(n)

  ci <- c(theta_hat - 1.96 * se_hat,
          theta_hat + 1.96 * se_hat)

  # -----------------------
  # 6) Return
  # -----------------------
  list(
    theta_hat = theta_hat,
    se_hat    = se_hat,
    ci        = ci,
    n         = n,
    K         = K,
    link      = link,
    learner   = learner,
    beta1_hat = beta1_hat,
    r_hat     = r_hat,
    mu_hat    = mu_hat,
    D_inv_hat = D_inv_hat,
    J_inv_hat = D_inv_hat
  )
}
