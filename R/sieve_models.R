# sieve_models.R
# Sieve / basis + glmnet nuisance learners for gvcm()
#' @noRd
NULL

# ------------------------------------------------------------
# Internal glmnet CV helper for sieve learner
# ------------------------------------------------------------

#' @noRd
.cv_glmnet_fit <- function(
    x, y, weights = NULL,
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

  args <- list(
    x = x,
    y = y,
    family = family,
    alpha = alpha,
    nfolds = nfolds,
    intercept = TRUE,
    standardize = FALSE
  )

  if (!is.null(weights)) {
    args$weights <- weights
  }

  fit <- do.call(glmnet::cv.glmnet, args)

  lam <- if (lambda_rule == "lambda.min") fit$lambda.min else fit$lambda.1se
  list(fit = fit, lambda = lam)
}

# ------------------------------------------------------------
# beta-model: fit sieve learner for
#   g(E[Y|X,Z]) = beta0(Z) + beta1(Z) X
# using design [B, X, B*X]
# ------------------------------------------------------------
#' @noRd
.fit_beta_sieve <- function(
    B_tr,
    X_tr,
    Y_tr,
    link,
    sieve_args
) {
  family_y <- match.arg(link, c("gaussian","binomial","poisson"))

  alpha <- sieve_args$alpha
  nfolds <- sieve_args$nfolds_glmnet
  lambda_rule <- sieve_args$lambda_rule

  XB_tr <- B_tr * as.numeric(X_tr)
  D_tr <- cbind(B_tr, X_tr, XB_tr)

  colnames(D_tr) <- c(
    paste0(colnames(B_tr), "_b0"),
    "X_main",
    paste0(colnames(B_tr), "_b1")
  )

  cv <- .cv_glmnet_fit(
    x = D_tr,
    y = Y_tr,
    family = family_y,
    alpha = alpha,
    nfolds = nfolds,
    lambda_rule = lambda_rule
  )

  list(
    cv = cv,
    pB = ncol(B_tr),
    colnames_B = colnames(B_tr)
  )
}


# ------------------------------------------------------------
# beta-model prediction:
# returns eta_hat and beta1_hat on test fold
# ------------------------------------------------------------
#' @noRd
.predict_beta_sieve <- function(
    fit,
    B_te,
    X_te
) {
  pB <- fit$pB

  XB_te <- B_te * as.numeric(X_te)
  D_te <- cbind(B_te, X_te, XB_te)

  eta_hat <- as.numeric(stats::predict(
    fit$cv$fit,
    newx = D_te,
    s = fit$cv$lambda,
    type = "link"
  ))

  coefs <- as.matrix(glmnet::coef.glmnet(
    fit$cv$fit$glmnet.fit,
    s = fit$cv$lambda
  ))

  # coefficient layout:
  # 1            : intercept
  # 2:(1+pB)     : beta0 basis coefficients
  # 2+pB         : X main effect
  # (3+pB):(2+2pB): beta1 basis coefficients
  idx_xmain <- 2 + pB
  idx_b1    <- (3 + pB):(2 + 2 * pB)

  x_main_coef <- coefs[idx_xmain, 1]
  beta1_coef  <- coefs[idx_b1, 1, drop = TRUE]

  beta1_hat <- as.numeric(x_main_coef + B_te %*% beta1_coef)

  list(
    eta_hat = eta_hat,
    beta1_hat = beta1_hat
  )
}


# ------------------------------------------------------------
# m-model: fit m(Z) = E[X | Z]
# ------------------------------------------------------------
#' @noRd
.fit_m_sieve <- function(
    B_tr,
    X_tr,
    sieve_args
) {
  alpha <- sieve_args$alpha
  nfolds <- sieve_args$nfolds_glmnet
  lambda_rule <- sieve_args$lambda_rule

  .cv_glmnet_fit(
    x = B_tr,
    y = as.numeric(X_tr),
    family = "gaussian",
    alpha = alpha,
    nfolds = nfolds,
    lambda_rule = lambda_rule
  )
}


# ------------------------------------------------------------
# m-model prediction
# ------------------------------------------------------------
#' @noRd
.predict_m_sieve <- function(
    fit,
    B_te
) {
  as.numeric(stats::predict(
    fit$fit,
    newx = B_te,
    s = fit$lambda,
    type = "response"
  ))
}


# ------------------------------------------------------------
# inverse-information model:
# weighted regression for the inverse quantity
# ------------------------------------------------------------
#' @noRd
.fit_invar_sieve <- function(
    B_tr,
    Y_tr,
    weights,
    sieve_args
) {
  alpha <- sieve_args$alpha
  nfolds <- sieve_args$nfolds_glmnet
  lambda_rule <- sieve_args$lambda_rule

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


# ------------------------------------------------------------
# inverse-information prediction
# ------------------------------------------------------------
#' @noRd
.predict_invar_sieve <- function(
    fit,
    B_te
) {
  as.numeric(stats::predict(
    fit$fit,
    newx = B_te,
    s = fit$lambda,
    type = "response"
  ))
}
