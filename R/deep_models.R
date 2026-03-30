# deepnet_models.R
# torch-based deepnet nuisance learners for gvcm()
#' @noRd
NULL

# ============================================================
# Helpers
# ============================================================

#' @noRd
.as_matrix_float <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}

#' @noRd
.get_device <- function(device = c("cpu", "cuda")) {
  device <- match.arg(device)
  if (device == "cuda" && torch::cuda_is_available()) {
    torch::torch_device("cuda")
  } else {
    torch::torch_device("cpu")
  }
}

#' @noRd
.beta_loss_from_eta <- function(eta, y, link = c("gaussian","binomial","poisson")) {
  link <- match.arg(link, c("gaussian","binomial","poisson"))

  if (link == "gaussian") {
    return(torch::nnf_mse_loss(eta, y))
  }

  if (link == "binomial") {
    # eta is the logit
    return(torch::nnf_binary_cross_entropy_with_logits(
      input = eta,
      target = y
    ))
  }

  # poisson with log link: eta = log(mu)
  # loss = mean(exp(eta) - y * eta), ignoring constant log(y!)
  eta_safe <- torch::torch_clamp(eta, min = -10, max = 10)
  mu <- torch::torch_exp(eta_safe)
  return(torch::torch_mean(mu - y * eta_safe))
}

#' @noRd
.make_activation <- function(name = c("silu", "relu", "gelu", "tanh")) {
  name <- match.arg(name)
  switch(
    name,
    silu = torch::nn_silu(),
    relu = torch::nn_relu(),
    gelu = torch::nn_gelu(),
    tanh = torch::nn_tanh()
  )
}

# ============================================================
# Residual block for MLP trunk
# ============================================================

#' @noRd
ResidualBlock <- torch::nn_module(
  "ResidualBlock",

  initialize = function(dim, dropout = 0, activation = "silu") {
    self$fc1 <- torch::nn_linear(dim, dim)
    self$fc2 <- torch::nn_linear(dim, dim)
    self$drop <- torch::nn_dropout(p = dropout)
    self$act <- .make_activation(activation)
  },

  forward = function(x) {
    out <- self$fc1(x)
    out <- self$act(out)
    out <- self$drop(out)
    out <- self$fc2(out)

    out <- out + x
    out <- self$act(out)
    out
  }
)
# ============================================================
# Two-head beta network
# shared trunk -> beta0(z), beta1(z)
# ============================================================

#' @noRd
BetaTwoHeadNet <- torch::nn_module(
  "BetaTwoHeadNet",

  initialize = function(
    p,
    hidden_dims = c(32),
    dropout = 0,
    n_residual = 1,
    activation = "silu"
  ) {
    stopifnot(length(hidden_dims) >= 1)

    self$input_layer <- torch::nn_linear(p, hidden_dims[1])
    self$act <- .make_activation(activation)
    self$drop <- torch::nn_dropout(p = dropout)

    self$hidden_layers <- torch::nn_module_list()
    if (length(hidden_dims) >= 2) {
      for (j in 2:length(hidden_dims)) {
        self$hidden_layers$append(
          torch::nn_linear(hidden_dims[j - 1], hidden_dims[j])
        )
      }
    }

    self$res_blocks <- torch::nn_module_list()
    last_dim <- hidden_dims[length(hidden_dims)]
    for (j in seq_len(n_residual)) {
      self$res_blocks$append(
        ResidualBlock(
          dim = last_dim,
          dropout = dropout,
          activation = activation
        )
      )
    }

    self$head_beta0 <- torch::nn_linear(last_dim, 1)
    self$head_beta1 <- torch::nn_linear(last_dim, 1)
  },

  forward = function(z) {
    h <- self$input_layer(z)
    h <- self$act(h)
    h <- self$drop(h)

    if (length(self$hidden_layers) > 0) {
      for (j in seq_len(length(self$hidden_layers))) {
        h <- self$hidden_layers[[j]](h)
        h <- self$act(h)
        h <- self$drop(h)
      }
    }

    if (length(self$res_blocks) > 0) {
      for (j in seq_len(length(self$res_blocks))) {
        h <- self$res_blocks[[j]](h)
      }
    }

    list(
      beta0 = self$head_beta0(h),
      beta1 = self$head_beta1(h)
    )
  }
)

#' @noRd
.fit_beta_deepnet <- function(
    Z_tr,
    X_tr,
    Y_tr,
    link,
    net_args
) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Package 'torch' is required for learner = 'deepnet'.")
  }

  link <- match.arg(link, c("gaussian","binomial","poisson"))

  defaults <- list(
    hidden_dims = c(32),
    dropout = 0,
    n_residual = 1,
    activation = "silu",
    lr = 5e-4,
    epochs = 200,
    batch_size = 32,
    weight_decay = 1e-4,
    lambda_beta1 = 1e-4,
    valid_prop = 0.1,
    early_stop_patience = 15,
    min_delta = 1e-4,
    device = "cpu",
    seed = 1,
    verbose = FALSE
  )

  net_args <- utils::modifyList(defaults, net_args)

  Z_tr <- .as_matrix_float(Z_tr)
  X_tr <- as.numeric(X_tr)
  Y_tr <- as.numeric(Y_tr)

  n <- nrow(Z_tr)
  p <- ncol(Z_tr)

  if (length(X_tr) != n || length(Y_tr) != n) {
    stop("Z_tr, X_tr, and Y_tr must have compatible lengths.")
  }

  set.seed(net_args$seed)
  torch::torch_manual_seed(net_args$seed)

  device <- .get_device(net_args$device)

  n_valid <- max(1L, floor(net_args$valid_prop * n))
  idx_all <- sample.int(n)
  idx_valid <- idx_all[seq_len(n_valid)]
  idx_train <- idx_all[-seq_len(n_valid)]

  # standardize X using training split only
  x_center <- mean(X_tr[idx_train])
  x_scale  <- stats::sd(X_tr[idx_train])
  if (!is.finite(x_scale) || x_scale <= 0) x_scale <- 1

  X_tr_std <- (X_tr - x_center) / x_scale

  z_train <- torch::torch_tensor(
    Z_tr[idx_train, , drop = FALSE],
    dtype = torch::torch_float(),
    device = device
  )
  x_train <- torch::torch_tensor(
    matrix(X_tr_std[idx_train], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )
  y_train <- torch::torch_tensor(
    matrix(Y_tr[idx_train], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )

  z_valid <- torch::torch_tensor(
    Z_tr[idx_valid, , drop = FALSE],
    dtype = torch::torch_float(),
    device = device
  )
  x_valid <- torch::torch_tensor(
    matrix(X_tr_std[idx_valid], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )
  y_valid <- torch::torch_tensor(
    matrix(Y_tr[idx_valid], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )

  model <- BetaTwoHeadNet(
    p = p,
    hidden_dims = net_args$hidden_dims,
    dropout = net_args$dropout,
    n_residual = net_args$n_residual,
    activation = net_args$activation
  )
  model$to(device = device)

  optimizer <- torch::optim_adam(
    params = model$parameters,
    lr = net_args$lr,
    weight_decay = net_args$weight_decay
  )

  best_state <- NULL
  best_valid <- Inf
  patience_count <- 0L

  batch_size <- min(net_args$batch_size, length(idx_train))
  n_train <- length(idx_train)

  for (epoch in seq_len(net_args$epochs)) {
    model$train()

    batch_order <- sample.int(n_train)
    batch_starts <- seq.int(1L, n_train, by = batch_size)

    epoch_loss <- 0

    for (b in batch_starts) {
      b_end <- min(b + batch_size - 1L, n_train)
      idx_b <- batch_order[b:b_end]

      zb <- z_train[idx_b, , drop = FALSE]
      xb <- x_train[idx_b, , drop = FALSE]
      yb <- y_train[idx_b, , drop = FALSE]

      optimizer$zero_grad()

      out <- model(zb)
      beta0_b <- out$beta0
      beta1_b <- out$beta1

      eta_b <- beta0_b + beta1_b * xb

      loss_outcome <- .beta_loss_from_eta(
        eta = eta_b,
        y = yb,
        link = link
      )

      penalty_beta1 <- net_args$lambda_beta1 * torch::torch_mean(beta1_b^2)
      loss <- loss_outcome + penalty_beta1

      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 5)
      optimizer$step()

      epoch_loss <- epoch_loss + loss$item()
    }

    model$eval()
    torch::with_no_grad({
      out_val <- model(z_valid)
      beta0_val <- out_val$beta0
      beta1_val <- out_val$beta1
      eta_val <- beta0_val + beta1_val * x_valid

      valid_outcome <- .beta_loss_from_eta(
        eta = eta_val,
        y = y_valid,
        link = link
      )

      valid_penalty <- net_args$lambda_beta1 * torch::torch_mean(beta1_val^2)
      valid_loss <- (valid_outcome + valid_penalty)$item()
    })

    if (isTRUE(net_args$verbose)) {
      message(
        sprintf(
          "beta deepnet epoch %d | train_loss = %.5f | valid_loss = %.5f",
          epoch, epoch_loss, valid_loss
        )
      )
    }

    if (valid_loss < best_valid - net_args$min_delta) {
      best_valid <- valid_loss
      best_state <- model$state_dict()
      patience_count <- 0L
    } else {
      patience_count <- patience_count + 1L
    }

    if (patience_count >= net_args$early_stop_patience) {
      if (isTRUE(net_args$verbose)) {
        message("beta deepnet early stopping triggered.")
      }
      break
    }
  }

  if (!is.null(best_state)) {
    model$load_state_dict(best_state)
  }

  list(
    model = model,
    device = device,
    link = link,
    net_args = net_args,
    p = p,
    x_center = x_center,
    x_scale = x_scale
  )
}

# ============================================================
# Predict beta deepnet
# Returns eta_hat and beta1_hat
# ============================================================

#' @noRd
.predict_beta_deepnet <- function(
    fit,
    Z_te,
    X_te
) {
  Z_te <- .as_matrix_float(Z_te)
  X_te <- as.numeric(X_te)

  if (nrow(Z_te) != length(X_te)) {
    stop("Z_te and X_te must have compatible lengths.")
  }

  z_te <- torch::torch_tensor(
    Z_te,
    dtype = torch::torch_float(),
    device = fit$device
  )

  fit$model$eval()

  out <- torch::with_no_grad({
    head_out <- fit$model(z_te)

    beta0_std <- as.numeric(head_out$beta0$to(device = torch::torch_device("cpu")))
    beta1_std <- as.numeric(head_out$beta1$to(device = torch::torch_device("cpu")))

    # back-transform to original X scale
    beta1 <- beta1_std / fit$x_scale
    beta0 <- beta0_std - beta1_std * fit$x_center / fit$x_scale

    eta <- beta0 + beta1 * X_te

    list(
      eta_hat = eta,
      beta1_hat = beta1,
      beta0_hat = beta0
    )
  })

  out
}

# ============================================================
# One-head regression network for m(z) = E[X | Z]
# ============================================================

#' @noRd
MRegressionNet <- torch::nn_module(
  "MRegressionNet",

  initialize = function(
    p,
    hidden_dims = c(32),
    dropout = 0,
    n_residual = 1,
    activation = "silu"
  ) {
    stopifnot(length(hidden_dims) >= 1)

    self$input_layer <- torch::nn_linear(p, hidden_dims[1])
    self$act <- .make_activation(activation)
    self$drop <- torch::nn_dropout(p = dropout)

    self$hidden_layers <- torch::nn_module_list()
    if (length(hidden_dims) >= 2) {
      for (j in 2:length(hidden_dims)) {
        self$hidden_layers$append(
          torch::nn_linear(hidden_dims[j - 1], hidden_dims[j])
        )
      }
    }

    last_dim <- hidden_dims[length(hidden_dims)]

    self$res_blocks <- torch::nn_module_list()
    for (j in seq_len(n_residual)) {
      self$res_blocks$append(
        ResidualBlock(
          dim = last_dim,
          dropout = dropout,
          activation = activation
        )
      )
    }

    self$head_m <- torch::nn_linear(last_dim, 1)
  },

  forward = function(z) {
    h <- self$input_layer(z)
    h <- self$act(h)
    h <- self$drop(h)

    if (length(self$hidden_layers) > 0) {
      for (j in seq_len(length(self$hidden_layers))) {
        h <- self$hidden_layers[[j]](h)
        h <- self$act(h)
        h <- self$drop(h)
      }
    }

    if (length(self$res_blocks) > 0) {
      for (j in seq_len(length(self$res_blocks))) {
        h <- self$res_blocks[[j]](h)
      }
    }

    self$head_m(h)
  }
)

# ============================================================
# Fit m deepnet
# ============================================================

#' @noRd
.fit_m_deepnet <- function(
    Z_tr,
    X_tr,
    net_args
) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Package 'torch' is required for learner = 'deepnet'.")
  }

  defaults <- list(
    hidden_dims = c(32),
    dropout = 0,
    n_residual = 1,
    activation = "silu",
    lr = 5e-4,
    epochs = 200,
    batch_size = 32,
    weight_decay = 1e-4,
    valid_prop = 0.1,
    early_stop_patience = 15,
    min_delta = 1e-4,
    device = "cpu",
    seed = 1,
    verbose = FALSE
  )

  net_args <- utils::modifyList(defaults, net_args)

  Z_tr <- .as_matrix_float(Z_tr)
  X_tr <- as.numeric(X_tr)

  n <- nrow(Z_tr)
  p <- ncol(Z_tr)

  if (length(X_tr) != n) {
    stop("Z_tr and X_tr must have compatible lengths.")
  }

  set.seed(net_args$seed)
  torch::torch_manual_seed(net_args$seed)

  device <- .get_device(net_args$device)

  n_valid <- max(1L, floor(net_args$valid_prop * n))
  idx_all <- sample.int(n)
  idx_valid <- idx_all[seq_len(n_valid)]
  idx_train <- idx_all[-seq_len(n_valid)]

  z_train <- torch::torch_tensor(
    Z_tr[idx_train, , drop = FALSE],
    dtype = torch::torch_float(),
    device = device
  )
  x_train <- torch::torch_tensor(
    matrix(X_tr[idx_train], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )

  z_valid <- torch::torch_tensor(
    Z_tr[idx_valid, , drop = FALSE],
    dtype = torch::torch_float(),
    device = device
  )
  x_valid <- torch::torch_tensor(
    matrix(X_tr[idx_valid], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )

  model <- MRegressionNet(
    p = p,
    hidden_dims = net_args$hidden_dims,
    dropout = net_args$dropout,
    n_residual = net_args$n_residual,
    activation = net_args$activation
  )
  model$to(device = device)

  optimizer <- torch::optim_adam(
    params = model$parameters,
    lr = net_args$lr,
    weight_decay = net_args$weight_decay
  )

  best_state <- NULL
  best_valid <- Inf
  patience_count <- 0L

  batch_size <- min(net_args$batch_size, length(idx_train))
  n_train <- length(idx_train)

  for (epoch in seq_len(net_args$epochs)) {
    model$train()

    batch_order <- sample.int(n_train)
    batch_starts <- seq.int(1L, n_train, by = batch_size)

    epoch_loss <- 0

    for (b in batch_starts) {
      b_end <- min(b + batch_size - 1L, n_train)
      idx_b <- batch_order[b:b_end]

      zb <- z_train[idx_b, , drop = FALSE]
      xb <- x_train[idx_b, , drop = FALSE]

      optimizer$zero_grad()

      pred_b <- model(zb)
      loss <- torch::nnf_mse_loss(pred_b, xb)

      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 5)
      optimizer$step()

      epoch_loss <- epoch_loss + loss$item()
    }

    model$eval()
    torch::with_no_grad({
      pred_val <- model(z_valid)
      valid_loss <- torch::nnf_mse_loss(pred_val, x_valid)$item()
    })

    if (isTRUE(net_args$verbose)) {
      message(
        sprintf(
          "m deepnet epoch %d | train_loss = %.5f | valid_loss = %.5f",
          epoch, epoch_loss, valid_loss
        )
      )
    }

    if (valid_loss < best_valid - net_args$min_delta) {
      best_valid <- valid_loss
      best_state <- model$state_dict()
      patience_count <- 0L
    } else {
      patience_count <- patience_count + 1L
    }

    if (patience_count >= net_args$early_stop_patience) {
      if (isTRUE(net_args$verbose)) {
        message("m deepnet early stopping triggered.")
      }
      break
    }
  }

  if (!is.null(best_state)) {
    model$load_state_dict(best_state)
  }

  list(
    model = model,
    device = device,
    net_args = net_args,
    p = p
  )
}

# ============================================================
# Predict m deepnet
# ============================================================

#' @noRd
.predict_m_deepnet <- function(
    fit,
    Z_te
) {
  Z_te <- .as_matrix_float(Z_te)

  z_te <- torch::torch_tensor(
    Z_te,
    dtype = torch::torch_float(),
    device = fit$device
  )

  fit$model$eval()

  out <- torch::with_no_grad({
    pred <- fit$model(z_te)
    as.numeric(pred$to(device = torch::torch_device("cpu")))
  })

  out
}

# ============================================================
# One-head regression network for inverse-information nuisance
# ============================================================

#' @noRd
InvarRegressionNet <- torch::nn_module(
  "InvarRegressionNet",

  initialize = function(
    p,
    hidden_dims = c(32),
    dropout = 0,
    n_residual = 1,
    activation = "silu"
  ) {
    stopifnot(length(hidden_dims) >= 1)

    self$input_layer <- torch::nn_linear(p, hidden_dims[1])
    self$act <- .make_activation(activation)
    self$drop <- torch::nn_dropout(p = dropout)

    self$hidden_layers <- torch::nn_module_list()
    if (length(hidden_dims) >= 2) {
      for (j in 2:length(hidden_dims)) {
        self$hidden_layers$append(
          torch::nn_linear(hidden_dims[j - 1], hidden_dims[j])
        )
      }
    }

    last_dim <- hidden_dims[length(hidden_dims)]

    self$res_blocks <- torch::nn_module_list()
    for (j in seq_len(n_residual)) {
      self$res_blocks$append(
        ResidualBlock(
          dim = last_dim,
          dropout = dropout,
          activation = activation
        )
      )
    }

    self$head_out <- torch::nn_linear(last_dim, 1)
  },

  forward = function(z) {
    h <- self$input_layer(z)
    h <- self$act(h)
    h <- self$drop(h)

    if (length(self$hidden_layers) > 0) {
      for (j in seq_len(length(self$hidden_layers))) {
        h <- self$hidden_layers[[j]](h)
        h <- self$act(h)
        h <- self$drop(h)
      }
    }

    if (length(self$res_blocks) > 0) {
      for (j in seq_len(length(self$res_blocks))) {
        h <- self$res_blocks[[j]](h)
      }
    }

    self$head_out(h)
  }
)

# ============================================================
# Weighted MSE loss
# ============================================================

#' @noRd
.weighted_mse_loss <- function(pred, target, weight) {
  torch::torch_mean(weight * (pred - target)^2)
}

# ============================================================
# Fit inverse-information deepnet
# ============================================================

#' @noRd
.fit_invar_deepnet <- function(
    Z_tr,
    Y_tr,
    weights,
    net_args
) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("Package 'torch' is required for learner = 'deepnet'.")
  }

  defaults <- list(
    hidden_dims = c(32),
    dropout = 0,
    n_residual = 1,
    activation = "silu",
    lr = 5e-4,
    epochs = 200,
    batch_size = 32,
    weight_decay = 1e-4,
    valid_prop = 0.1,
    early_stop_patience = 15,
    min_delta = 1e-4,
    device = "cpu",
    seed = 1,
    verbose = FALSE
  )

  net_args <- utils::modifyList(defaults, net_args)

  Z_tr <- .as_matrix_float(Z_tr)
  Y_tr <- as.numeric(Y_tr)
  weights <- as.numeric(weights)

  n <- nrow(Z_tr)
  p <- ncol(Z_tr)

  if (length(Y_tr) != n || length(weights) != n) {
    stop("Z_tr, Y_tr, and weights must have compatible lengths.")
  }

  if (any(!is.finite(Y_tr)) || any(!is.finite(weights))) {
    stop("Y_tr and weights must be finite.")
  }

  if (any(weights < 0)) {
    stop("weights must be nonnegative.")
  }

  set.seed(net_args$seed)
  torch::torch_manual_seed(net_args$seed)

  device <- .get_device(net_args$device)

  n_valid <- max(1L, floor(net_args$valid_prop * n))
  idx_all <- sample.int(n)
  idx_valid <- idx_all[seq_len(n_valid)]
  idx_train <- idx_all[-seq_len(n_valid)]

  z_train <- torch::torch_tensor(
    Z_tr[idx_train, , drop = FALSE],
    dtype = torch::torch_float(),
    device = device
  )
  y_train <- torch::torch_tensor(
    matrix(Y_tr[idx_train], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )
  w_train <- torch::torch_tensor(
    matrix(weights[idx_train], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )

  z_valid <- torch::torch_tensor(
    Z_tr[idx_valid, , drop = FALSE],
    dtype = torch::torch_float(),
    device = device
  )
  y_valid <- torch::torch_tensor(
    matrix(Y_tr[idx_valid], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )
  w_valid <- torch::torch_tensor(
    matrix(weights[idx_valid], ncol = 1),
    dtype = torch::torch_float(),
    device = device
  )

  model <- InvarRegressionNet(
    p = p,
    hidden_dims = net_args$hidden_dims,
    dropout = net_args$dropout,
    n_residual = net_args$n_residual,
    activation = net_args$activation
  )
  model$to(device = device)

  optimizer <- torch::optim_adam(
    params = model$parameters,
    lr = net_args$lr,
    weight_decay = net_args$weight_decay
  )

  best_state <- NULL
  best_valid <- Inf
  patience_count <- 0L

  batch_size <- min(net_args$batch_size, length(idx_train))
  n_train <- length(idx_train)

  for (epoch in seq_len(net_args$epochs)) {
    model$train()

    batch_order <- sample.int(n_train)
    batch_starts <- seq.int(1L, n_train, by = batch_size)

    epoch_loss <- 0

    for (b in batch_starts) {
      b_end <- min(b + batch_size - 1L, n_train)
      idx_b <- batch_order[b:b_end]

      zb <- z_train[idx_b, , drop = FALSE]
      yb <- y_train[idx_b, , drop = FALSE]
      wb <- w_train[idx_b, , drop = FALSE]

      optimizer$zero_grad()

      pred_b <- model(zb)
      loss <- .weighted_mse_loss(
        pred = pred_b,
        target = yb,
        weight = wb
      )

      loss$backward()
      torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = 5)
      optimizer$step()

      epoch_loss <- epoch_loss + loss$item()
    }

    model$eval()
    torch::with_no_grad({
      pred_val <- model(z_valid)
      valid_loss <- .weighted_mse_loss(
        pred = pred_val,
        target = y_valid,
        weight = w_valid
      )$item()
    })

    if (isTRUE(net_args$verbose)) {
      message(
        sprintf(
          "invar deepnet epoch %d | train_loss = %.5f | valid_loss = %.5f",
          epoch, epoch_loss, valid_loss
        )
      )
    }

    if (valid_loss < best_valid - net_args$min_delta) {
      best_valid <- valid_loss
      best_state <- model$state_dict()
      patience_count <- 0L
    } else {
      patience_count <- patience_count + 1L
    }

    if (patience_count >= net_args$early_stop_patience) {
      if (isTRUE(net_args$verbose)) {
        message("invar deepnet early stopping triggered.")
      }
      break
    }
  }

  if (!is.null(best_state)) {
    model$load_state_dict(best_state)
  }

  list(
    model = model,
    device = device,
    net_args = net_args,
    p = p
  )
}

# ============================================================
# Predict inverse-information deepnet
# Returns q_hat(Z), not yet divided by V(mu)
# ============================================================

#' @noRd
.predict_invar_deepnet <- function(
    fit,
    Z_te
) {
  Z_te <- .as_matrix_float(Z_te)

  z_te <- torch::torch_tensor(
    Z_te,
    dtype = torch::torch_float(),
    device = fit$device
  )

  fit$model$eval()

  out <- torch::with_no_grad({
    pred <- fit$model(z_te)
    as.numeric(pred$to(device = torch::torch_device("cpu")))
  })

  out
}
