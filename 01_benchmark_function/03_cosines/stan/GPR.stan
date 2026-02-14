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
}


transformed data {
    real y_mean;
    vector[N] y_avezero;
    real delta = 1e-9;
    vector[N] mu;
    for (i in 1:N) mu[i] = 0;
    y_mean = mean(y);
    y_avezero = y - y_mean;
}

parameters {
    vector<lower=0>[D] rho;
    real<lower=0> alpha;
    real<lower=0> sd_e;
}

model {
    matrix[N, N] K;

    K = L_cov_exp_quad_ARD(X, alpha, rho, N, delta, sd_e);
    alpha ~ std_normal();
    rho ~ inv_gamma(5, 5);
    sd_e ~ normal(0, 1);

    y_avezero ~ multi_normal(mu, K);

}

generated quantities {
    vector[N_grid] y_new;
    matrix[N+1, N+1] K;
    matrix[N, N] K11;
    vector[N] K12;
    real K22;
    real mu_star;
    real sigma;
    matrix[N+1,D] X_x;

for (n in 1:N_grid) {
    X_x[1:N,] = X;
    X_x[N+1,] = X_grid[n,];

    K = L_cov_exp_quad_ARD(X_x, alpha, rho, N+1, delta, sd_e);
    K11 = K[1:N,1:N];
    K12 = K[1:N,N+1];
    K22 = K[N+1,N+1];

    mu_star = K12' * inverse(K11) * y_avezero;

    sigma = K22 - K12' * inverse(K11) * K12;
    sigma = sigma^0.5;
    y_new[n] = normal_rng(mu_star, sigma) + y_mean;
}
}
