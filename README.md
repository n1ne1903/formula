
# 📚 Tổng Hợp Công Thức NLP, Machine Learning, Deep Learning & Computer Vision

---

## 🧠 NLP – Natural Language Processing

### 📊 TF-IDF
$$
TF(t, d) = \frac{\text{số lần t xuất hiện trong d}}{\text{tổng số từ trong d}}, \quad
IDF(t) = \log \left( \frac{N}{\text{số văn bản chứa t}} \right)
$$
$$
TFIDF(t, d) = TF(t, d) \cdot IDF(t)
$$

### 📐 Cosine Similarity
$$
\text{CosineSim}(A, B) = \frac{A \cdot B}{\|A\| \cdot \|B\|}
$$

### 📉 Cross Entropy Loss
$$
\mathcal{L} = - \sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

### 🔄 Perplexity
$$
\text{Perplexity} = \exp\left( - \frac{1}{N} \sum_{i=1}^N \log P(w_i|w_{<i}) \right)
$$

### 🧩 Attention (Transformer)
$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

### 🧬 Positional Encoding
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$

### 🧠 F1 Score
$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

---

## 🤖 Machine Learning

### 📉 Linear Regression
$$
\hat{y} = w^T x + b, \quad
\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### 🔢 Logistic Regression
$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad
\hat{y} = \sigma(w^T x + b)
$$
$$
\mathcal{L}_{BCE} = - \left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

### 🌳 Decision Tree
$$
Entropy = - \sum_{i=1}^k p_i \log_2 p_i, \quad
IG(D, A) = H(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} H(D_v)
$$

### 🔍 KNN
$$
d(x, x') = \sqrt{\sum_{i=1}^n (x_i - x_i')^2}
$$

### 🚀 Gradient Descent
$$
w := w - \eta \cdot \nabla_w \mathcal{L}
$$

### 💨 Momentum
$$
v_t = \gamma v_{t-1} + \eta \nabla_w \mathcal{L}, \quad w := w - v_t
$$

### 📦 RMSProp
$$
s_t = \rho s_{t-1} + (1 - \rho)(\nabla_w \mathcal{L})^2, \quad
w := w - \frac{\eta}{\sqrt{s_t + \epsilon}} \nabla_w \mathcal{L}
$$

### 📏 Metrics
$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}, \quad
Precision = \frac{TP}{TP + FP}, \quad
Recall = \frac{TP}{TP + FN}
$$
$$
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
$$

### 🔄 PCA
$$
\Sigma = \frac{1}{n} X^T X, \quad Z = X \cdot W_k
$$

### 🧠 SVM
$$
f(x) = w^T x + b, \quad y_i(w^T x_i + b) \geq 1, \quad \text{minimize } \frac{1}{2} \|w\|^2
$$

---

## 🧱 Computer Vision

### 📐 Conv2D Output
$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - f}{s} + 1 \right\rfloor, \quad
W_{out} = \left\lfloor \frac{W_{in} + 2p - f}{s} + 1 \right\rfloor
$$

### 📦 Conv Parameters
$$
Params = (f \cdot f \cdot C_{in}) \cdot C_{out} + C_{out}
$$

### 📉 Pooling Output
$$
H_{out} = \left\lfloor \frac{H_{in} - f}{s} + 1 \right\rfloor
$$

### 📏 IoU
$$
IoU = \frac{B_1 \cap B_2}{B_1 \cup B_2}
$$

### 🧭 Geometric Transformations
**Translation**:
$$
x' = x + dx, \quad y' = y + dy
$$

**Scaling**:
$$
x' = s_x x, \quad y' = s_y y
$$

**Rotation**:
$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
$$

### 🧮 Euclidean Distance
$$
d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$
