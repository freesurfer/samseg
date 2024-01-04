import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from numba import njit, prange
from numba.typed import List


@njit
def py_fast_digamma(x):
    "Faster digamma function assumes x > 0."
    r = 0
    while x <= 5:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    t = f * (
        -1 / 12.0
        + f
        * (
            1 / 120.0
            + f
            * (
                -1 / 252.0
                + f
                * (
                    1 / 240.0
                    + f
                    * (
                        -1 / 132.0
                        + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617 / 8160.0))
                    )
                )
            )
        )
    )
    return r + np.log(x) - 0.5 / x + t


@njit
def _compute_q_labeling_and_mu_s_slow(
    data,
    fixed_priors,
    q_phi,
    sigma_d,
    q_mu_s,
    q_sigma_s,
    q_mu_0k,
    lambda_k,
    fat_shift,
    alpha_k,
    beta_k,
    fat_start_index,
):
    # As the approximate label posterior depends on neighboring voxels,
    # namely those that are either at position i-fat_shift or i+fat_shift,
    # we need to iterate the updates in batches. Here I'll assume that the border
    # is fixed and do this slice-by-slice not to mess up.
    # We can of course do this in batches to speed it up.
    slices = list(range(fat_shift, data.shape[0] - fat_shift))
    # random.shuffle(slices)
    for i in slices:
        # First update q_phi
        phi_new_i = np.log(fixed_priors[i, :, :, :] + 1e-15)
        for k in range(fixed_priors.shape[-1]):
            phi_new_i[:, :, k] -= 0.5 * (
                np.log(beta_k[k]) - py_fast_digamma(alpha_k[k])
            ) + (alpha_k[k] / (2 * beta_k[k])) * (
                q_sigma_s[i, :, :]
                + q_mu_s[i, :, :] ** 2
                - 2 * q_mu_s[i, :, :] * q_mu_0k[k]
                + beta_k[k] / (lambda_k[k] * (alpha_k[k] - 1))
                + q_mu_0k[k] ** 2
            )

        # Loop through the fat (only one) and water classes
        q_phi_w = np.sum(q_phi[i + fat_shift][:, :, :fat_start_index], axis=2)

        q_mu_i_s = q_mu_s[i, :, :]
        q_mu_shifted_down_s = q_mu_s[i + fat_shift, :, :]
        for k in range(fat_start_index, len(lambda_k)):
            phi_new_i[:, :, k] += (1 / (sigma_d)) * (
                q_mu_i_s * (data[i + fat_shift, :, :] - q_phi_w * q_mu_shifted_down_s)
            )

        q_phi_f = np.sum(q_phi[i - fat_shift][:, :, fat_start_index:], axis=2)
        q_mu_shifted_up_s = q_mu_s[i - fat_shift, :, :]
        for k in range(fat_start_index):
            phi_new_i[:, :, k] += (1 / (sigma_d)) * (
                q_mu_i_s * (data[i, :, :] - q_phi_f * q_mu_shifted_up_s)
            )

        exp_sum_tmp_i = -100000 * np.ones(phi_new_i.shape[:-1])
        exp_sum_tmp_i = exp_sum_tmp_i.reshape(np.prod(np.array(exp_sum_tmp_i.shape)))
        for k in range(phi_new_i.shape[-1]):
            phi_new_i_tmp = np.ascontiguousarray(phi_new_i[:, :, k])
            phi_new_i_tmp = phi_new_i_tmp.reshape(
                np.prod(np.array(phi_new_i.shape[:-1]))
            )
            max_mask = phi_new_i_tmp > exp_sum_tmp_i
            larger_values = phi_new_i_tmp[max_mask]
            exp_sum_tmp_i[max_mask] = larger_values

        exp_sum_tmp_i = exp_sum_tmp_i.reshape(phi_new_i.shape[:-1])

        tmp_exp_sum = np.zeros_like(exp_sum_tmp_i)
        for k in range(q_phi.shape[-1]):
            tmp_exp_sum += np.exp(phi_new_i[:, :, k] - exp_sum_tmp_i)

        normalizer = exp_sum_tmp_i + np.log(tmp_exp_sum)

        for k in range(q_phi.shape[-1]):
            phi_new_i[:, :, k] = np.exp(phi_new_i[:, :, k] - normalizer)

        # Update phi on this slice
        q_phi[i, :, :, :] = phi_new_i
        # Then update mu_s on this slice
        mu_s_new_i = np.zeros(phi_new_i.shape[:-1])
        for k in range(q_phi.shape[-1]):
            mu_s_new_i += (
                sigma_d * q_phi[i, :, :, k] * q_mu_0k[k] * alpha_k[k]
            ) / beta_k[k]

        q_phi_f_shifted_up = np.sum(
            q_phi[i - fat_shift][:, :, fat_start_index:], axis=2
        )

        mu_s_shifted_up = q_mu_s[i - fat_shift, :, :]
        q_phi_w = np.sum(q_phi[i][:, :, :fat_start_index], axis=2)
        mu_s_new_i += q_phi_w * (data[i, :, :] - q_phi_f_shifted_up * mu_s_shifted_up)

        q_phi_f = np.sum(q_phi[i][:, :, fat_start_index:], axis=2)
        q_phi_w_shifted_down = np.sum(
            q_phi[i + fat_shift][:, :, :fat_start_index], axis=2
        )

        mu_s_shifted_down = q_mu_s[i + fat_shift, :, :]
        mu_s_new_i += q_phi_f * (
            data[i + fat_shift, :, :] - q_phi_w_shifted_down * mu_s_shifted_down
        )

        tmp = np.zeros(phi_new_i.shape[:-1])
        for k in range(q_phi.shape[-1]):
            tmp += (sigma_d * q_phi[i, :, :, k] * alpha_k[k]) / beta_k[k]

        mu_s_new_i /= 1 + tmp
        q_mu_s[i, :, :] = mu_s_new_i

    return q_phi, q_mu_s


def _compute_q_sigma_s(phi, sigma_d, alpha_k, beta_k):
    # This doesn't need padding either as we only need
    # to know phi at the given location
    tmp = 0
    for k in range(phi.shape[-1]):
        tmp += (sigma_d * phi[:, :, :, k] * alpha_k[k]) / beta_k[k]

    sigma_s = (sigma_d) / (1 + tmp)
    return sigma_s


def _compute_q_mu_and_sigma_k(phi, mu_s, sigma_s):

    lambda_k = List([0.0] * phi.shape[-1])
    alpha_k = lambda_k.copy()
    beta_k = lambda_k.copy()
    mu_0k = lambda_k.copy()

    for k in range(phi.shape[-1]):
        tmp = np.sum(phi[:, :, :, k].flatten())
        lambda_k[k] = tmp
        alpha_k[k] = tmp / 2

    for k in range(phi.shape[-1]):
        mu_0k[k] = np.sum(phi[:, :, :, k].flatten() * mu_s.flatten()) / np.sum(
            phi[:, :, :, k].flatten()
        )
        beta_k[k] = (
            np.sum(phi[:, :, :, k].flatten() * (mu_s.flatten() + sigma_s.flatten()))
            / 2
            * np.sum(phi[:, :, :, k].flatten())
        )
        beta_k[k] -= (mu_0k[k] ** 2) / 2

    return mu_0k, lambda_k, alpha_k, beta_k


@njit(parallel=True)
def _elbo(
    lambda_k,
    sigma_s,
    phi,
    priors,
    alpha_k,
    beta_k,
    mu_0k,
    mu_s,
    fat_shift,
    sigma_d,
    d,
    fat_start_index,
):

    dim = phi.shape[:-1]
    elbo = 0.5 * sum(1 + np.log(sigma_s.flatten()))
    elbo -= sum(phi.flatten() * np.log(phi.flatten() + 1e-15))
    elbo += sum(phi.flatten() * np.log(priors.flatten() + 1e-15))

    tmp = 0
    for k in prange(len(lambda_k)):
        tmp += (
            -0.5 * np.log(lambda_k[k])
            + (3.0 / 2.0) * np.log(beta_k[k])
            - (alpha_k[k] + 3.0 / 2.0) * py_fast_digamma(alpha_k[k])
            + alpha_k[k] ** 2 / (2 * (alpha_k[k] - 1))
        )

    elbo -= tmp

    tmp = np.zeros(dim)
    for k in prange(len(lambda_k)):
        phi_tmp = phi[:, :, :, k]
        tmp += (phi_tmp * alpha_k[k] / (2 * beta_k[k])) * (
            sigma_s
            + mu_s**2
            - 2 * mu_s * mu_0k[k]
            + beta_k[k] / (lambda_k[k] * (alpha_k[k] - 1))
            + mu_0k[k] ** 2
        )

    elbo -= np.sum(tmp.flatten())

    for k in prange(len(alpha_k)):
        phi_tmp = phi[:, :, :, k]
        tmp += phi_tmp * (
            np.log(2 * np.pi) + np.log(beta_k[k]) - py_fast_digamma(alpha_k[k])
        )

    elbo -= 0.5 * np.sum(tmp.flatten())

    tmp = np.zeros(dim)
    phi_f = np.sum(phi[:, :, :, fat_start_index:], axis=3)
    phi_w = np.sum(phi[:, :, :, :fat_start_index], axis=3)
    for i in prange(fat_shift, d.shape[0]):
        tmp += (
            -2
            * d[i, :]
            * (
                phi_w[i, :, :] * mu_s[i, :, :]
                + phi_f[i - fat_shift, :, :] * mu_s[i - fat_shift, :, :]
            )
            + phi_w[i, :, :] * (sigma_s[i, :, :] + mu_s[i, :, :] ** 2)
            + 2
            * phi_w[i, :, :]
            * phi_f[i - fat_shift, :, :]
            * mu_s[i, :]
            * mu_s[i - fat_shift, :, :]
            + phi_f[i - fat_shift, :, :]
            * (sigma_s[i - fat_shift, :, :] + mu_s[i - fat_shift, :, :] ** 2)
        )

    for i in prange(fat_shift):
        tmp += -2 * d[i, :, :] * (phi_w[i, :, :] * mu_s[i, :, :]) + phi_w[i, :, :] * (
            sigma_s[i, :, :] + mu_s[i, :, :] ** 2
        )

    tmp = np.sum(tmp.flatten())
    elbo -= 0.5 * (1 / sigma_d) * tmp

    return elbo


# _mu_s(image), _mu_0k (mean), _sigma_0k(variance)
class GMM_fat_shift:
    # from _prepare_data(...)
    def __init__(self, fat_shift, sigma_d,
                 numberOfGaussiansPerClass, classNames, initialWs,
                 classPriors, gaussianPosteriors, imageBuffer, mask):

        # class variables:
        #   _fat_shift, _sigma_d
        #   _imagedata
        #   _fixed_priors, _fixed_posteriors
        #   _fat_gaussians, _water_gaussians
        #   _fat_start_index
        #   _phi, _mu_s, _sigma_s, _mu_0k, _lambda_k, _alpha_k, _beta_k

        self._fat_shift = fat_shift
        self._sigma_d = sigma_d
        
        nGaussiansPerClass = numberOfGaussiansPerClass

        dim = imageBuffer.shape
        priors = np.zeros((*dim, classPriors.shape[-1]))
        posteriors = np.zeros((*dim, gaussianPosteriors.shape[-1]))

        for i in range(classPriors.shape[-1]):
            priors[mask, i] = classPriors[:, i]

        for i in range(gaussianPosteriors.shape[-1]):
            posteriors[mask, i] = gaussianPosteriors[:, i]        

        # Extract a 2d slice and prepare the priors for running the variational inference
        self._imagedata = np.squeeze(imageBuffer[:, ::-1, :])
        priors = np.squeeze(priors[:, ::-1, :, :])
        posteriors = np.squeeze(posteriors[:, ::-1, :, :])
    
        # Reshuffle the classes so that the fat classes are in the end
        class_permutation = []
        fat_classes = []
        for i, s in enumerate(classNames):
            if "Fatty" in s or "Spongy" in s:
                fat_classes.append(i)
            else:
                class_permutation.append(i)

        class_permutation += fat_classes
        # re-order priors after reshuffling classes
        priors = priors[:, :, :, class_permutation]

        # reshuffle the gaussians so that fat gaussians are in the end
        gaussian_permutation = []
        fat_gaussians = []
        for i, (g, n) in enumerate(zip(nGaussiansPerClass, classNames)):
            for component in range(g):
                gaussian_number = sum(nGaussiansPerClass[:i]) + component
                if "Fatty" in n or "Spongy" in n:
                    fat_gaussians.append(gaussian_number)
                else:
                    gaussian_permutation.append(gaussian_number)

        # re-number the gaussians
        water_gaussians = np.arange(len(gaussian_permutation)).tolist()
        fat_gaussians_tmp = np.arange(
            len(gaussian_permutation), len(gaussian_permutation) + len(fat_gaussians)
        ).tolist()

        gaussian_permutation += fat_gaussians

        self._water_gaussians = water_gaussians
        self._fat_gaussians = fat_gaussians_tmp

        # re-order ws, nGaussiansPerClass, posteriors after reshuffling gaussians
        ws = np.array(initialWs)[gaussian_permutation].tolist()
        nGaussiansPerClass = np.array(nGaussiansPerClass)[
            class_permutation
        ].tolist()
        posteriors = posteriors[:, :, :, gaussian_permutation]
        #self.classNames = [classNames[i] for i in class_permutation]

        priors_split = np.zeros_like(posteriors)
        for class_number in range(len(nGaussiansPerClass)):
            prior = priors[:, :, :, class_number]
            number_of_components = nGaussiansPerClass[class_number]
            for component_number in range(number_of_components):
                gaussian_number = (
                    sum(nGaussiansPerClass[:class_number]) + component_number
                )
                priors_split[:, :, :, gaussian_number] = ws[gaussian_number] * prior

        priors = priors_split

        self._imagedata = np.moveaxis(self._imagedata, 1, 0)
        priors = np.moveaxis(priors, 1, 0)
        posteriors = np.moveaxis(posteriors, 1, 0)
        self._imagedata = zoom(self._imagedata, [2, 1, 1], order=1)

        # fixed priors and posteriors
        self._fixed_priors = zoom(priors,[2, 1, 1, 1], order=1)
        self._fixed_posteriors = zoom(posteriors,[2, 1, 1,1], order=1)

        self._fat_start_index = len(water_gaussians)


    # from _intialize_model_parameters(priors, posteriors, data)
    def initializeGMMParameters(self):

        # Let's create some initial values.
        # The means are based on soft weighting of the data with the estimated posteriors
        self._mu_s = np.zeros(self._imagedata.shape)

        # Note that the posteriors are shifted already as we estimated those without accounting for the shift
        # Actually so are the priors but let's see maybe we can snap out of that
        for k in range(self._fixed_posteriors.shape[-1]):
            self._mu_s += np.squeeze(self._fixed_posteriors[:, :, :, k]) * self._imagedata

        # Let's init the sigma_s to just some fixed value
        self._sigma_s = 2 * np.ones_like(self._imagedata)

        # And let's set _phi equal to the priors initially
        self._phi = self._fixed_posteriors.copy()

        self._mu_0k, self._lambda_k, self._alpha_k, self._beta_k = _compute_q_mu_and_sigma_k(self._phi, self._mu_s, self._sigma_s)

        return self._phi, self._mu_s, self._sigma_s, self._mu_0k, self._lambda_k, self._alpha_k, self._beta_k
        

    def fitGMMParameters(self, niter):
        for i in range(niter):
            # approximate distribution of labels
            (self._phi, self._mu_s) = _compute_q_labeling_and_mu_s_slow(
                self._imagedata,
                self._fixed_priors,
                self._phi,
                self._sigma_d,
                self._mu_s,
                self._sigma_s,
                self._mu_0k,
                self._lambda_k,
                self._fat_shift,
                self._alpha_k,
                self._beta_k,
                self._fat_start_index,
            )
            
            # approximate distribution for the non-shifted signal
            self._sigma_s = _compute_q_sigma_s(self._phi, self._sigma_d, self._alpha_k, self._beta_k)

        # approximate distribution for the mixture parameters means and variances
        self._mu_0k, self._lambda_k, self._alpha_k, self._beta_k = _compute_q_mu_and_sigma_k(self._phi, self._mu_s, self._sigma_s)
        
        return self._phi, self._mu_s, self._sigma_s, self._mu_0k, self._lambda_k, self._alpha_k, self._beta_k

        
    def evaluateELBO(self):
        elbo = _elbo(
            self._lambda_k,
            self._sigma_s,
            self._phi,
            self._fixed_priors,
            self._alpha_k,
            self._beta_k,
            self._mu_0k,
            self._mu_s,
            self._fat_shift,
            self._sigma_d,
            self._imagedata,
            self._fat_start_index,
        )
        return elbo

    
    @property
    def imagedata(self):
        return self._imagedata

    @property
    def fixed_priors(self):
        return self._fixed_priors

    @property
    def fixed_posteriors(self):
        return self._fixed_posteriors

    @property
    def fat_gaussians(self):
        return self._fat_gaussians

    @property
    def water_gaussians(self):
        return self._water_gaussians

    @property
    def phi(self):
        return self._phi

    @property
    def mu_s(self):
        return self._mu_s

    @property
    def sigma_s(self):
        return self._sigma_s

    @property
    def mu_0k(self):
        return self._mu_0k

    @property
    def lambda_k(self):
        return self._lambda_k

    @property
    def alpha_k(self):
        return self._alpha_k

    @property
    def beta_k(self):
        return self._beta_k
