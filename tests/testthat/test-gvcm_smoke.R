test_that("gvcm runs for each link with sieve learner", {
  skip_if_not_installed("glmnet")

  sim_gvcm_data <- function(n = 300,
                            link = c("gaussian", "binomial", "poisson"),
                            seed = 1) {
    link <- match.arg(link)
    set.seed(seed)

    Z <- matrix(rnorm(n * 3), n, 3)
    colnames(Z) <- paste0("Z", 1:3)

    z1 <- Z[, 1]
    z2 <- Z[, 2]
    z3 <- Z[, 3]

    beta0 <- 0.2 + 0.3 * sin(z1) + 0.2 * z2
    beta1 <- 0.5 + 0.3 * cos(z1) - 0.2 * z3
    pX <- plogis(-0.2 + 0.6 * z1 - 0.4 * z2)
    X <- rbinom(n, 1, pX)

    eta <- beta0 + beta1 * X

    if (link == "gaussian") Y <- eta + rnorm(n)
    if (link == "binomial") Y <- rbinom(n, 1, plogis(eta))
    if (link == "poisson")  Y <- rpois(n, exp(eta))

    data.frame(Y = Y, X = X, Z)
  }

  for (lk in c("gaussian", "binomial", "poisson")) {
    dat <- sim_gvcm_data(link = lk, seed = 1)

    fit <- gvcm(
      data = dat,
      Y = "Y",
      X = "X",
      Z = paste0("Z", 1:3),
      link = lk,
      learner = "sieve",
      K = 3,
      sieve_args = list(
        basis = "ns",
        df = 3,
        alpha = 1,
        nfolds_glmnet = 3,
        lambda_rule = "lambda.1se"
      )
    )

    expect_true(is.list(fit))
    expect_true(is.finite(fit$theta_hat))
    expect_true(is.finite(fit$se_hat))
    expect_true(length(fit$ci) == 2)
    expect_true(all(is.finite(fit$ci)))
    expect_identical(fit$link, lk)
    expect_identical(fit$learner, "sieve")
  }
})

test_that("gvcm runs for each link with deepnet learner", {
  skip_if_not_installed("torch")

  sim_gvcm_data <- function(n = 300,
                            link = c("gaussian", "binomial", "poisson"),
                            seed = 1) {
    link <- match.arg(link)
    set.seed(seed)

    Z <- matrix(rnorm(n * 3), n, 3)
    colnames(Z) <- paste0("Z", 1:3)

    z1 <- Z[, 1]
    z2 <- Z[, 2]
    z3 <- Z[, 3]

    beta0 <- 0.2 + 0.3 * sin(z1) + 0.2 * z2
    beta1 <- 0.5 + 0.3 * cos(z1) - 0.2 * z3
    pX <- plogis(-0.2 + 0.6 * z1 - 0.4 * z2)
    X <- rbinom(n, 1, pX)

    eta <- beta0 + beta1 * X

    if (link == "gaussian") Y <- eta + rnorm(n)
    if (link == "binomial") Y <- rbinom(n, 1, plogis(eta))
    if (link == "poisson")  Y <- rpois(n, exp(eta))

    data.frame(Y = Y, X = X, Z)
  }

  for (lk in c("gaussian", "binomial", "poisson")) {
    dat <- sim_gvcm_data(link = lk, seed = 1)

    fit <- gvcm(
      data = dat,
      Y = "Y",
      X = "X",
      Z = paste0("Z", 1:3),
      link = lk,
      learner = "deepnet",
      K = 3,
      net_args = list(
        hidden_dims = c(16),
        dropout = 0,
        n_residual = 1,
        lr = 5e-4,
        epochs = 30,
        batch_size = 32,
        weight_decay = 1e-4,
        valid_prop = 0.1,
        early_stop_patience = 5,
        min_delta = 1e-4,
        device = "cpu",
        seed = 1,
        verbose = FALSE
      )
    )

    expect_true(is.finite(fit$theta_hat))
    expect_true(is.finite(fit$se_hat))
    expect_true(length(fit$ci) == 2)
    expect_true(all(is.finite(fit$ci)))
    expect_identical(fit$link, lk)
    expect_identical(fit$learner, "deepnet")
  }
})
