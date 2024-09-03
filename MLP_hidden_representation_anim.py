import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

#DATA 

def spiral_data():
    data = []
    labels = []
    n = 100

    # Generate first spiral
    for i in range(n):
        r = i/n * 5 + np.random.uniform(-0.1, 0.1)
        t = 1.25 * i/n * 2 * np.pi + np.random.uniform(-0.1, 0.1)
        data.append([r * np.sin(t), r * np.cos(t)])
        labels.append(1)

    # Generate second spiral
    for i in range(n):
        r = i/n * 5 + np.random.uniform(-0.1, 0.1)
        t = 1.25 * i/n * 2 * np.pi + np.pi + np.random.uniform(-0.1, 0.1)
        data.append([r * np.sin(t), r * np.cos(t)])
        labels.append(0)

    N = len(data)
    return data, labels

def circle_data():
    data = []
    labels = []

    # Generate points for the first circle
    for i in range(50):
        r = np.random.uniform(0.0, 2.0)
        t = np.random.uniform(0.0, 2 * np.pi)
        data.append([r * np.sin(t), r * np.cos(t)])
        labels.append(1)

    # Generate points for the second circle
    for i in range(50):
        r = np.random.uniform(3.0, 5.0)
        t = 2 * np.pi * i / 50.0
        data.append([r * np.sin(t), r * np.cos(t)])
        labels.append(0)

    N = len(data)
    return data, labels



data, labels = circle_data()
# data, labels = spiral_data()


# create training and testing data from the data
data = np.array(data)
labels = np.array(labels)
N = len(data)
idx = np.arange(N)
np.random.shuffle(idx)
data = data[idx]
labels = labels[idx]

# Split data into training and testing
split = int(0.8 * N)
train_data = data[:split]
train_labels = labels[:split]
test_data = data[split:]
test_labels = labels[split:]

# Plot the spiral data
# plt.scatter(data[:,0], data[:,1], c=labels, cmap='coolwarm')
# plt.show()

# Convert data to PyTorch tensors
train_data = torch.FloatTensor(train_data)
train_labels = torch.LongTensor(train_labels)
test_data = torch.FloatTensor(test_data)
test_labels = torch.LongTensor(test_labels)

# Create a model
class Model(torch.nn.Module):
    x_before_output = None

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(2, 3)
        self.fc2 = torch.nn.Linear(3, 3)
        self.fc3 = torch.nn.Linear(3, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Instantiate the model, the loss function, and the optimizer using basic functions
model = Model()
# Cross-entropy loss
criterion = torch.nn.CrossEntropyLoss()
# SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
n_epochs = 1500
batch_size = 10
train_losses = []
test_losses = []

for epoch in tqdm(range(n_epochs)):
    model.train()
    permutation = torch.randperm(train_data.size()[0])

    for i in range(0, train_data.size()[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch_x, batch_y = train_data[indices], train_labels[indices]

        optimizer.zero_grad()
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        permutation = torch.randperm(test_data.size()[0])
        for i in range(0, test_data.size()[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = test_data[indices], test_labels[indices]

            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)
            test_losses.append(loss.item())

# Calculate the accuracy
model.eval()
y_pred = model(test_data)
y_pred = torch.argmax(y_pred, dim=1)
accuracy = (test_labels == y_pred).sum().item() / test_labels.size(0)
print(f'Accuracy: {accuracy:.2f}')

# data after each layer
x = torch.FloatTensor(data)
first_layer = model.fc1(x)
first_layer_relu = torch.relu(first_layer)
second_layer = model.fc2(first_layer_relu)
second_layer_relu = torch.relu(second_layer)
output_layer = model.fc3(second_layer_relu)

#activations of hidden layer
hidden_activations = second_layer.detach().numpy()

# Plot input layer representation (train + test) [2d]
#plot the data
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', label='Train')
plt.legend()
plt.savefig('input_layer.png')

# CREATE ANIMATION
# Plot second hidden layer representation (train + test) [3d]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def init():
    ax.scatter(hidden_activations[:, 0], hidden_activations[:, 1], hidden_activations[:, 2], c=labels, cmap='coolwarm', label='Train')
    return fig,

def animate(i):
    ax.view_init(azim=i)
    return fig,

# Animate
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=360, interval=20, blit=True)
# Save
anim.save('hidden_layer_representation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])