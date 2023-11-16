import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


df_test = pd.read_csv('mnist_test.csv')


X_test = df_test.iloc[:, :-1].values


scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)


X_test_tensor = torch.Tensor(X_test)

class PrimeModel(nn.Module):
    def __init__(self, input_size):
        super(PrimeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

#
model_path = 'prime_model.pth'
input_size = X_test.shape[1]  # Giriş boyutunu belirle
loaded_model = PrimeModel(input_size)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()  # Modeli değerlendirme moduna ayarla

df_test = pd.read_csv('mnist_test.csv')
X_test = df_test.iloc[:, :-1].values
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
X_test_tensor = torch.Tensor(X_test)

with torch.no_grad():
    test_outputs = loaded_model(X_test_tensor)
    predicted_labels = (test_outputs >= 0.5).float()

df_test['Predicted'] = predicted_labels.numpy().flatten()


print(df_test[['label', 'Predicted']])

df_test[['label', 'Predicted']].to_csv('predicted_results.csv', index=False)
