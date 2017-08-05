import matplotlib.pyplot as plt
import numpy as np



# prior_variances = np.log([.001, .01, .1, 1., 10., 100., 1000.])#
# train_accs = [.801, .942, .976, .982, .980, .979, .979]
# test_accs = [.816, .938, .966, .968, .970, .965, .970]

# no_pW_train = [.979,.979,.979,.979,.979,.979,.979]
# no_pW_test = [.970,.970,.970,.970,.970,.970,.970]





# fig = plt.figure(facecolor='white')
# ax = fig.add_subplot(111, frameon=False)
# # ax.axis('off')
# # ax.set_yticks([])
# # ax.set_xticks([])
# plt.plot(prior_variances, no_pW_train, label='No p(W) train')
# plt.plot(prior_variances, no_pW_test, label='No p(W) test')
# plt.plot(prior_variances, train_accs, label='Train')
# plt.plot(prior_variances, test_accs, label='Test')



# ax.set_ylabel('Accuracy')
# ax.legend(fontsize=9, loc=0)
# plt.show()





prior_variances = np.log([.001, .01, .1, 1., 10.])#
elbo = [-83, -6, -147, -365, -593]
# test_accs = [.816, .938, .966, .968, .970, .965, .970]

# no_pW_train = [.979,.979,.979,.979,.979,.979,.979]
# no_pW_test = [.970,.970,.970,.970,.970,.970,.970]





fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111, frameon=False)
# ax.axis('off')
# ax.set_yticks([])
# ax.set_xticks([])
# plt.plot(prior_variances, no_pW_train, label='No p(W) train')
# plt.plot(prior_variances, no_pW_test, label='No p(W) test')
# plt.plot(prior_variances, train_accs, label='Train')
plt.plot(prior_variances, elbo)#, label='ELBO')



ax.set_ylabel('ELBO')
ax.legend(fontsize=9, loc=0)
plt.show()













