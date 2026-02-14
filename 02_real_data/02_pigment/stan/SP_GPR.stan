functions {
    matrix L_cov_exp_quad_ARD(matrix x,
                            real alpha,
                            vector rho,
                            int N,
                              real delta,
                              real sd_e) {
    matrix[N, N] K;
    real sq_alpha = square(alpha);
        real sq_sd_e = square(sd_e);
    for (i in 1:(N-1)) {
            K[i, i] = sq_alpha + delta + sq_sd_e;
        for (j in (i + 1):N) {
            K[i, j] = sq_alpha* exp(-0.5 * dot_self((x[i]' - x[j]') ./ rho));
            K[j, i] = K[i, j];
      }
    }
        K[N, N] = sq_alpha + delta + sq_sd_e;

    return K;
  }
}

data {
    int<lower=1> N;
    int<lower=1> D;
    matrix[N,D] X;
    vector[N] y;
    int<lower=1> N_grid;
    matrix[N_grid,D] X_grid;
    vector[D] a_mean;
    vector[D] a_sd;
    real b_mean;
    real b_sd;
}

transformed data {
    real delta = 1e-9;
}

parameters {
    vector<lower=0>[D] rho;
    real<lower=0> alpha;
    real<lower=0> sd_e;
    real a1;
    real a2;
    real a3;
    real a4;
    real a5;
    real a6;
    real b;
}

model {
    vector[N] mu;
    vector[D] a;
    matrix[N, N] K;

    a[1] = a1;
    a[2] = a2;
    a[3] = a3;
    a[4] = a4;
    a[5] = a5;
    a[6] = a6;

    K = L_cov_exp_quad_ARD(X, alpha, rho, N, delta, sd_e);

    alpha ~ std_normal();
    rho ~ inv_gamma(5, 5);
    sd_e ~ normal(0, 1);
    a1 ~ normal(a_mean[1], a_sd[1]);
    a2 ~ normal(a_mean[2], a_sd[2]);
    a3 ~ normal(a_mean[3], a_sd[3]);
    a4 ~ normal(a_mean[4], a_sd[4]);
    a5 ~ normal(a_mean[5], a_sd[5]);
    a6 ~ normal(a_mean[6], a_sd[6]);
    b ~ normal(b_mean, b_sd);
    mu = X*a + b;

    y ~ multi_normal(mu, K);
}

generated quantities {
  vector[N_grid] y_new;
  vector[D] a;
  vector[N] y_Gx;
  matrix[N+1,D] X_x;
  matrix[N+1, N+1] K;
  matrix[N, N] K11;
  vector[N] K12;
  real K22;
  real mu_star1;
  real mu_star2;
  real mu_star;
  real sigma;

  a[1] = a1;
  a[2] = a2;
  a[3] = a3;
  a[4] = a4;
  a[5] = a5;
  a[6] = a6;
  y_Gx = y - (X*a + b);

  for (n in 1:N_grid) {
    X_x[1:N,] = X;
    X_x[N+1,] = X_grid[n,];
    K = L_cov_exp_quad_ARD(X_x, alpha, rho, N+1, delta, sd_e);
    K11 = K[1:N,1:N];
    K12 = K[1:N,N+1];
    K22 = K[N+1,N+1];

    mu_star1 = X_grid[n,]*a + b;
    mu_star2 = K12' * inverse(K11) * y_Gx;
    mu_star = mu_star1 + mu_star2;
    sigma = K22 - K12' * inverse(K11) * K12;
    sigma = sigma^0.5;
    y_new[n] = normal_rng(mu_star, sigma);
  }
}
