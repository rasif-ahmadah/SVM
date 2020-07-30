import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# men-generate dataset menggunakan make_moons dari sklearn
pos, neg = make_moons(n_samples=100, noise=0.1)

# Split data menjadi data train (80%) dan data test (20%)
pos_train, pos_test, neg_train, neg_test = train_test_split(
    pos, neg, test_size=0.2, random_state=0)

# memberi warna data (train & test) berdasarkan kelas
plt.scatter(pos_train[:, 0], pos_train[:, 1], c=ListedColormap(('red', 'green'))
            (neg_train), s=30, cmap=plt.cm.Paired)
plt.scatter(pos_test[:, 0], pos_test[:, 1], c=ListedColormap(('blue', 'yellow'))
            (neg_test), s=30, cmap=plt.cm.Paired)

# membuat SVM classifier gaussian dengan C = 1000 & gamma = 0.1
classifier = svm.SVC(kernel='rbf', C=1000, gamma=0.1)

# mengklasifikasikan data train
classifier.fit(pos_train, neg_train)

# membuat mesh untuk plot dengan step 0.02
# margin / jarak
step = .02
x_min, x_max = pos[:, 0].min() - 1, pos[:, 0].max() + 1
y_min, y_max = pos[:, 1].min() - 1, pos[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))

# ploting decision boundary (hyperplane) dan margin
xy = np.vstack([xx.ravel(), yy.ravel()]).T
Z = classifier.decision_function(xy).reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.75,
            linestyles=['--', '-', '--'])

# plot support vector
plt.scatter(classifier.support_vectors_[:, 0], classifier.support_vectors_[:, 1],
            s=100, facecolors='k')

plt.axis('tight')

# classification report
pred = classifier.predict(pos_test)
print(classification_report(neg_test, pred))

plt.show()
