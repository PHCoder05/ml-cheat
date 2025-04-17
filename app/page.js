"use client";
import { useState } from "react";
import { Copy, Check } from "lucide-react";

const algorithms = [
  {
    id: 1,
    name: "svm classification",
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
    name: "birthweight dataset analysis",
    code: `...your code here...`,
  },
  {
    id: 3,
    name: "logistic regression",
    code: `...your code here...`,
  },
  {
    id: 4,
    name: "pca visualization",
    code: `...your code here...`,
  },
  {
    id: 5,
    name: "decision tree classification",
    code: `...your code here...`,
  },
  {
    id: 6,
    name: "k-means clustering",
    code: `...your code here...`,
  },
  {
    id: 7,
    name: "random forest model",
    code: `...your code here...`,
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
        <h2 className="text-sm font-semibold mb-3 text-gray-700">Practicals</h2>
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
        Select a practical and copy the code using the button.
      </div>
    </div>
  );
}
