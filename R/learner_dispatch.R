# learner_dispatch.R
#' @noRd
NULL

#' @noRd
.fit_beta_model <- function(
    learner,
    B_tr,
    Z_tr,
    X_tr,
    Y_tr,
    link,
    sieve_args,
    net_args
) {
  learner <- match.arg(learner, c("sieve", "deepnet"))

  if (learner == "sieve") {
    return(
      .fit_beta_sieve(
        B_tr = B_tr,
        X_tr = X_tr,
        Y_tr = Y_tr,
        link = link,
        sieve_args = sieve_args
      )
    )
  }

  if (learner == "deepnet") {
    return(
      .fit_beta_deepnet(
        Z_tr = Z_tr,
        X_tr = X_tr,
        Y_tr = Y_tr,
        link = link,
        net_args = net_args
      )
    )
  }
}

#' @noRd
.predict_beta_model <- function(
    fit,
    learner,
    B_te,
    Z_te,
    X_te
) {
  learner <- match.arg(learner, c("sieve", "deepnet"))

  if (learner == "sieve") {
    return(
      .predict_beta_sieve(
        fit = fit,
        B_te = B_te,
        X_te = X_te
      )
    )
  }

  if (learner == "deepnet") {
    return(
      .predict_beta_deepnet(
        fit = fit,
        Z_te = Z_te,
        X_te = X_te
      )
    )
  }
}

#' @noRd
.fit_m_model <- function(
    learner,
    B_tr,
    Z_tr,
    X_tr,
    sieve_args,
    net_args
) {
  learner <- match.arg(learner, c("sieve", "deepnet"))

  if (learner == "sieve") {
    return(
      .fit_m_sieve(
        B_tr = B_tr,
        X_tr = X_tr,
        sieve_args = sieve_args
      )
    )
  }

  if (learner == "deepnet") {
    return(
      .fit_m_deepnet(
        Z_tr = Z_tr,
        X_tr = X_tr,
        net_args = net_args
      )
    )
  }
}

#' @noRd
.predict_m_model <- function(
    fit,
    learner,
    B_te,
    Z_te
) {
  learner <- match.arg(learner, c("sieve", "deepnet"))

  if (learner == "sieve") {
    return(
      .predict_m_sieve(
        fit = fit,
        B_te = B_te
      )
    )
  }

  if (learner == "deepnet") {
    return(
      .predict_m_deepnet(
        fit = fit,
        Z_te = Z_te
      )
    )
  }
}

#' @noRd
.fit_invar_model <- function(
    learner,
    B_tr,
    Z_tr,
    Y_tr,
    weights,
    sieve_args,
    net_args
) {
  learner <- match.arg(learner, c("sieve", "deepnet"))

  if (learner == "sieve") {
    return(
      .fit_invar_sieve(
        B_tr = B_tr,
        Y_tr = Y_tr,
        weights = weights,
        sieve_args = sieve_args
      )
    )
  }

  if (learner == "deepnet") {
    return(
      .fit_invar_deepnet(
        Z_tr = Z_tr,
        Y_tr = Y_tr,
        weights = weights,
        net_args = net_args
      )
    )
  }
}

#' @noRd
.predict_invar_model <- function(
    fit,
    learner,
    B_te,
    Z_te
) {
  learner <- match.arg(learner, c("sieve", "deepnet"))

  if (learner == "sieve") {
    return(
      .predict_invar_sieve(
        fit = fit,
        B_te = B_te
      )
    )
  }

  if (learner == "deepnet") {
    return(
      .predict_invar_deepnet(
        fit = fit,
        Z_te = Z_te
      )
    )
  }
}
