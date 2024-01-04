import numpy as np
#import nibabel as nib
#import matplotlib.pyplot as plt

from .GMM_fat_shift import GMM_fat_shift


class Fatshift:
    def __init__(self, fat_shift, sigma_d,
                 numberOfGaussiansPerClass, classNames, initialWs,
                 classPriors, classPosteriors, imageBuffer, mask):
        
        self.numberOfGaussiansPerClass = numberOfGaussiansPerClass
        self.classNames = classNames
        self.ws = initialWs
        self.classPriors = classPriors
        self.classPosteriors = classPosteriors
        self.imageBuffer = imageBuffer
        self.mask = mask

        self.fat_shift = fat_shift  # 3
        self.sigma_d = sigma_d  # 1
        self.gmm_fat_shift = None


    def fitModel(self):
        self.gmm_fat_shift = GMM_fat_shift(self.fat_shift, self.sigma_d,
                                           self.numberOfGaussiansPerClass,
                                           self.classNames,
                                           self.ws,
                                           self.classPriors,
                                           self.classPosteriors,
                                           self.imageBuffer,
                                           self.mask)
        
        self.estimateModelParameters()


    # iteration loop
    def estimateModelParameters(self):
        phi, mu_s, sigma_s, mu_0k, lambda_k, alpha_k, beta_k = self.gmm_fat_shift.initializeGMMParameters()

        iters = [1] * 10 + [1] * 20

        old_elbo = self.gmm_fat_shift.evaluateELBO()

        fixed_posteriors = self.gmm_fat_shift.fixed_posteriors
        imagedata = self.gmm_fat_shift.imagedata

        """
        fat_gaussians = self.gmm_fat_shift.fat_gaussians
        water_gaussians = self.gmm_fat_shift.water_gaussians
        
        post_f = np.sum(fixed_posteriors[:, :, :, fat_gaussians], axis=3)
        post_f = np.squeeze(post_f[:, 150, :])
        phi_f = np.squeeze(np.sum(phi[:, :, :, fat_gaussians], axis=3))
        phi_w = np.sum(phi[:, :, :, water_gaussians], axis=3)
        fig, axs = plt.subplots(3, 3)

        im10 = axs[1,0].imshow(np.squeeze(phi_w[:, 150, :]))
        im11 = axs[1,1].imshow(np.squeeze(phi[:, 150, :, fat_gaussians[0]]))
        im12 = axs[1,2].imshow(np.squeeze(phi[:, 150, :, fat_gaussians[1]]))
        im00 = axs[0,0].imshow(np.exp(np.squeeze(imagedata[:, 150, :])), cmap="gray")
        im01 = axs[0,1].imshow(np.exp(np.squeeze(mu_s[:, 150, :])), cmap="gray")
        post_rgb = np.zeros((post_f.shape[0], post_f.shape[1], 3))
        post_rgb[:, :, 0] = post_f
        post_rgb[:, :, 1] = np.squeeze(phi_f[:, 150, :])
        im02 = axs[0,2].imshow(post_rgb)
        axs[2,2].plot(0, 0, 'o')
        print('before pause')
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)
        print('after pause')
        """
        
        iters_elbo = []
        elbo_diff = []
        for j, niter in enumerate(iters):
            print(niter)
            
            """
            phi_f = np.squeeze(np.sum(phi[:, :, :, fat_gaussians], axis=3))
            phi_w = np.sum(phi[:, :, :, water_gaussians], axis=3)
            im10.set_data(np.squeeze(phi_w[:,150,:]))
            im11.set_data(np.squeeze(phi[:, 150, :, fat_gaussians[0]]))
            im12.set_data(np.squeeze(phi[:, 150, :, fat_gaussians[1]]))
            im00.set_data(np.exp(np.squeeze(imagedata[:, 150, :])))
            im01.set_data(np.exp(np.squeeze(mu_s[:, 150, :])))

            post_rgb = np.zeros((post_f.shape[0], post_f.shape[1], 3))
            post_rgb[:, :, 0] = post_f
            post_rgb[:, :, 1] = np.squeeze(phi_f[:, 150, :])
            im02.set_data(post_rgb)
            """

            print('blaa')
            phi, mu_s, sigma_s, mu_0k, lambda_k, alpha_k, beta_k = self.gmm_fat_shift.fitGMMParameters(niter)
            
            new_elbo = self.gmm_fat_shift.evaluateELBO()            
            iters_elbo.append(j)
            print("ELBO: " + str(old_elbo - new_elbo))
            elbo_diff.append(old_elbo-new_elbo)

            """
            axs[2,2].plot(iters_elbo, elbo_diff)
            """
            
            print(mu_0k)
            print(lambda_k)
            print(alpha_k)
            print(beta_k)
            old_elbo = np.copy(new_elbo)

            """
            fig.canvas.draw()
            fig.canvas.flush_events()
            """


        
