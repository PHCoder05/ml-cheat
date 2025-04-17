"use client";
import { useState } from "react";
import { Copy, Check } from "lucide-react";

const algorithms = [
  {
    id: 1,
    name: "svm",
    code: `from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_classifier = SVC(kernel='linear')

svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of the SVM classifier: {accuracy * 100:.2f}%")
`,
  },
  {
    id: 2,
    name: "bw",
    code: `import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

file_path = '/content/birthwt.csv'

df = pd.read_csv(file_path)

print(df.head())


age_corr, _ = pearsonr(df['age'], df['bwt'])
print(f'Correlation between Age and Birth Weight: {age_corr}')

X_age = sm.add_constant(df['age'])
y = df['bwt']

model_age = sm.OLS(y, X_age).fit()
print(model_age.summary())

lwt_corr, _ = pearsonr(df['lwt'], df['bwt'])
print(f'Correlation between Mother\'s Weight and Birth Weight: {lwt_corr}')

X_lwt = sm.add_constant(df['lwt'])
model_lwt = sm.OLS(y, X_lwt).fit()
print(model_lwt.summary())

plt.figure(figsize=(8, 6))
plt.scatter(df['age'], df['bwt'], color='blue', alpha=0.5)
plt.title('Mother\'s Age vs Birth Weight')
plt.xlabel('Mother\'s Age')
plt.ylabel('Birth Weight')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['lwt'], df['bwt'], color='green', alpha=0.5)
plt.title('Mother\'s Weight vs Birth Weight')
plt.xlabel('Mother\'s Weight')
plt.ylabel('Birth Weight')
plt.grid(True)
plt.show()
`,
  },
  {
    id: 3,
    name: "kM",
    code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

plt.figure(figsize=(8, 6))

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o')

plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')

plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

df = pd.DataFrame(X_scaled, columns=iris.feature_names)
df['Cluster'] = labels
print(df.head())
`,
  },
  {
    id: 4,
    name: "pca visualization",
    code: `...your code here...`,
  },
  {
    id: 5,
    name: "market ",
    code: `...your code here...`,
  },
  {
    id: 6,
    name: "NN",
    code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

iris = load_iris()
X = iris.data
y = iris.target

encoder = LabelBinarizer()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes in Iris

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=8)

plot_model(model, to_file='iris_model_architecture.png', show_shapes=True, show_layer_names=True)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
`,
  },
  {
    id: 7,
    name: "OCR",
    code: `import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))  # hidden layer
model.add(Dense(64, activation='relu'))   # another hidden layer
model.add(Dense(10, activation='softmax'))  # output layer (10 digits)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train_cat, epochs=5, validation_data=(x_test, y_test_cat))

loss, accuracy = model.evaluate(x_test, y_test_cat)
print(f"Test Accuracy: {accuracy:.4f}")

sample_index = 9
plt.imshow(x_test[sample_index], cmap='gray')
plt.title(f"Actual Label: {y_test[sample_index]}")
plt.show()

prediction = model.predict(np.expand_dims(x_test[sample_index], axis=0))
predicted_label = np.argmax(prediction)
print(f"Predicted Label: {predicted_label}")
`,
  },
  {
    id: 8,
    name: "GMM PCA",
    code: `# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = iris.data  
y = iris.target  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels, cmap='viridis', s=50)
plt.title("Clustering using GMM (PCA visualization)", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label='Cluster')

cluster_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

for i in range(3):
    plt.scatter(X_pca[gmm_labels == i, 0], X_pca[gmm_labels == i, 1], label=cluster_names[i])

plt.legend()
plt.show()
`,
  },

  // Add more if needed
];

export default function PracticalListFixedLeft() {
  const [copiedId, setCopiedId] = useState(null);

  const handleCopy = async (code, id) => {
    await navigator.clipboard.writeText(code);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  return (
    <div className="flex w-full min-h-screen bg-gray-50">
      {/* Left sidebar with all practicals */}
      <div className="w-64 h-screen overflow-y-auto sticky top-0 bg-white border-r shadow-sm p-4">
        <ul className="space-y-2 text-[10px] text-gray-700">
          {algorithms.map((algo) => (
            <li key={algo.id} className="flex justify-between items-center">
              <span className="truncate w-44">
                {algo.id}. {algo.name}
              </span>
              <button
                onClick={() => handleCopy(algo.code, algo.id)}
                className="text-gray-500 hover:text-black"
                title="Copy code">
                {copiedId === algo.id ? (
                  <Check size={12} />
                ) : (
                  <Copy size={12} />
                )}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Right side empty or for future use */}
      <div className="flex-1 p-8 flex justify-center items-center text-gray-400 text-sm">
      </div>
    </div>
  );
}
