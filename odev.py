import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

# XOR veri seti
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# Ağ boyutları
input_size = 2
hidden_size = 2
output_size = 1

# Ağırlıklar
w1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

w2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

lr = 0.1
epochs = 10000

# Eğitim Döngüsü (Tamamen sessiz çalışır)
for epoch in range(epochs):

    # İleri Besleme
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2) + b2
    y_pred = sigmoid(z2)

    # Hata
    error = y_pred - y

    # Geri Yayılım
    delta2 = error * sigmoid_derivative(z2)
    delta1 = np.dot(delta2, w2.T) * sigmoid_derivative(z1)

    # Gradyan Hesaplama
    dw2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)

    dw1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # Ağırlık Güncelleme
    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

# --- EĞİTİM SONU HESAPLAMALARI ---
# Ağırlıkların son haliyle tahminleri alıyoruz
z1 = np.dot(X, w1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, w2) + b2
pred = sigmoid(z2)

# Final hata oranını (Ortalama Karesel Hata - MSE) hesaplıyoruz
final_error = pred - y
final_mse = np.mean(np.square(final_error))

print(f"Eğitim Tamamlandı. Nihai Hata Oranı (MSE): {final_mse:.6f}\n")

print("--- Tahmin Sonuçları ---")
for i in range(len(X)):
    print(f"Girdi: {X[i]} -> Beklenen: {y[i][0]} | Ağın Çıktısı: {pred[i][0]:.4f} (Yuvarlanmış: {int(pred[i][0].round())})")