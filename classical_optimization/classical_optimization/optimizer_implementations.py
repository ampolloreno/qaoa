"""
A collection of (reference) optimizer implementations.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# Taken from the Google implementation.
def _get_least_squares_model_gradient(
        xs,
        ys,
        xopt):
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('linear_model', LinearRegression(fit_intercept=False)),
    ])
    shifted_xs = [x - xopt for x in xs]
    model = model.fit(shifted_xs, ys)
    fitted_coeffs = model.named_steps['linear_model'].coef_
    n = len(xs[0])
    linear_coeffs = np.array(fitted_coeffs[1:n + 1])
    return linear_coeffs, model


def uniformly_sample_from_box(center, radius, num_samples):
    """
    Sample num_samples points from a box around center with side length 2r.

    :param center: The center of the box to sample from, (x_0,..., x_{n-1})
    :param radius: The region in each parameter to consider, r. i.e. (x_0\pm r, ..., x_{n-1}\pm r)
    :param num_samples: The number of samples to draw.
    """
    samples = []
    for i in range(num_samples):
        sample = []
        for _ in center:
            sample.append(2 * (np.random.rand() - .5) + radius)
        samples.append(np.array(sample))
    return np.array(samples)


def model_gradient_descent(func, initial_point, sample_radius, neighborhood_sample_number, learning_rate,
                           stability_constant, decay_rate_exponent, tolerance, max_evals):
    x = initial_point
    function_evaluations = []
    number_of_evaluations = 0
    iteration = 0
    while number_of_evaluations + neighborhood_sample_number <= max_evals:  # We won't exceed the maximum sample number.
        iteration += 1
        samples = uniformly_sample_from_box(x, sample_radius, neighborhood_sample_number)
        sample_radius = sample_radius / iteration**decay_rate_exponent
        # Evaluate random samples in neighborhood.
        for sample in samples:
            function_evaluations.append((sample, func(sample)))
        all_samples_in_neighbourhood = []

        # Find all nearby samples.
        for sample in function_evaluations:
            if sum(np.abs(sample[0] - x)) < len(x) * sample_radius:
                all_samples_in_neighbourhood.append(sample)
        # Taken from Google's code.
        model_gradient, model = _get_least_squares_model_gradient(
            [sample[0] for sample in all_samples_in_neighbourhood],
            [sample[1] for sample in all_samples_in_neighbourhood],
            x)

        # calculate the gradient and update the current point
        gradient_norm = np.linalg.norm(model_gradient)
        decayed_rate = (
                learning_rate / (number_of_evaluations + 1 + stability_constant) ** decay_rate_exponent)
        # Convergence criteria
        if decayed_rate * gradient_norm < tolerance:
            converged = True
            message = 'Optimization converged successfully.'
            break
        # Update
        x -= decayed_rate * model_gradient
    return x