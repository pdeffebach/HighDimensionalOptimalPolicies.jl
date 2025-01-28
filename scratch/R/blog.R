# The goal pdf we hope to be able to sample
# from
target <- function(x) {
  exp(-x^2) * (2 + sin(5 * x) + sin(2 * x))
}

# Given a candidate MC (characterized by sigma)
# and a current value of x_n, return a one-row data frame
# for the value and if it was accepted
metropolis_step <- function(x, sigma) {
  proposed_x <- rnorm(1, mean = x, sd = sigma)
  accept_prob <- min(1, target(proposed_x) / target(x))
  u <- runif(1)
  if(u <= accept_prob) {
    value <- proposed_x
    accepted <- TRUE
  } else {
    value <- x
    accepted <- FALSE
  }
  out <- data.frame(value = value, accepted = accepted)
  out
}

# Record steps in the MC algorithm
# for a long length of time
# * initial_value: Starting position
# * n: Number of steps in the algorithm
# * sigma: Characteristic of candidate MC
# * burnin = 0: Number of iterations before
#   recording results
# * lag: Number of iterations we run the sampler
#   for between successive samples
metropolis_sampler <- function(
  initial_value, 
  n = 1000, 
  sigma = 1, 
  burnin = 0, 
  lag = 1) {

  results <- list()
  current_state <- initial_value
  for(i in 1:burnin) {
    out <- metropolis_step(current_state, sigma)
    current_state <- out$value
  }
  for(i in 1:n) {
    for(j in 1:lag) {
      out <- metropolis_step(current_state, sigma)
      current_state <- out$value
    }
    results[[i]] <- out
  }
  results <- do.call(rbind, results)
  results
}