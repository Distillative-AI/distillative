### Knowledge Distillation (Distillative AI)

1. **Paper: "Distilling the Knowledge in a Neural Network" (Hinton, Vinyals, Dean, 2015)**
   - **Summary:**
     - This foundational paper introduces the concept of knowledge distillation. The authors propose using a large, complex model (teacher) to train a smaller, more efficient model (student). The student model is trained on the soft targets provided by the teacher, which are the probability distributions over the classes.
   - **Key Contributions:**
     - The idea of using soft targets, which contain more information than hard labels (0s and 1s).
     - A technique to combine the distillation loss with the traditional loss function.
   - **Implementation:**
     ```python
     import torch
     import torch.nn as nn
     import torch.optim as optim

     class DistillationLoss(nn.Module):
         def __init__(self, teacher_model, temperature):
             super(DistillationLoss, self).__init__()
             self.teacher_model = teacher_model
             self.temperature = temperature
             self.kl_div = nn.KLDivLoss(reduction='batchmean')

         def forward(self, student_logits, target):
             with torch.no_grad():
                 teacher_logits = self.teacher_model(target)
             soft_target = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
             student_soft_target = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
             distillation_loss = self.kl_div(student_soft_target, soft_target) * (self.temperature ** 2)
             return distillation_loss
     ```

2. **Paper: "Do Deep Nets Really Need to be Deep?" (Ba, Caruana, 2014)**
   - **Summary:**
     - This paper explores the possibility of shallow networks achieving performance comparable to deep networks through distillation.
   - **Key Contributions:**
     - Demonstrated that shallow networks can often perform as well as deep networks when trained with knowledge distillation.
   - **Implementation:**
     ```python
     import torch.nn.functional as F

     def train_student(student_model, teacher_model, train_loader, optimizer, temperature, alpha):
         teacher_model.eval()
         student_model.train()
         criterion = nn.CrossEntropyLoss()

         for data, target in train_loader:
             optimizer.zero_grad()
             student_output = student_model(data)
             with torch.no_grad():
                 teacher_output = teacher_model(data)
             loss = alpha * criterion(student_output, target) + \
                    (1 - alpha) * F.kl_div(F.log_softmax(student_output / temperature, dim=1),
                                           F.softmax(teacher_output / temperature, dim=1),
                                           reduction='batchmean') * (temperature ** 2)
             loss.backward()
             optimizer.step()
     ```

### Generative AI

1. **Paper: "Generative Adversarial Nets" (Goodfellow et al., 2014)**
   - **Summary:**
     - This groundbreaking paper introduces GANs, where two neural networks (a generator and a discriminator) are trained in a competitive setting.
   - **Key Contributions:**
     - The introduction of adversarial training.
     - Demonstrated the potential of GANs in generating realistic images.
   - **Implementation:**
     ```python
     import torch
     import torch.nn as nn
     import torch.optim as optim

     class Generator(nn.Module):
         def __init__(self):
             super(Generator, self).__init__()
             self.main = nn.Sequential(
                 nn.Linear(100, 256),
                 nn.ReLU(True),
                 nn.Linear(256, 512),
                 nn.ReLU(True),
                 nn.Linear(512, 1024),
                 nn.ReLU(True),
                 nn.Linear(1024, 784),
                 nn.Tanh()
             )

         def forward(self, x):
             return self.main(x)

     class Discriminator(nn.Module):
         def __init__(self):
             super(Discriminator, self).__init__()
             self.main = nn.Sequential(
                 nn.Linear(784, 1024),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Linear(1024, 512),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Linear(512, 256),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Linear(256, 1),
                 nn.Sigmoid()
             )

         def forward(self, x):
             return self.main(x)

     def train_gan(generator, discriminator, data_loader, num_epochs, lr):
         criterion = nn.BCELoss()
         optimizer_g = optim.Adam(generator.parameters(), lr=lr)
         optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

         for epoch in range(num_epochs):
             for real_data, _ in data_loader:
                 batch_size = real_data.size(0)

                 # Train discriminator
                 real_labels = torch.ones(batch_size, 1)
                 fake_labels = torch.zeros(batch_size, 1)
                 outputs = discriminator(real_data)
                 d_loss_real = criterion(outputs, real_labels)

                 z = torch.randn(batch_size, 100)
                 fake_data = generator(z)
                 outputs = discriminator(fake_data.detach())
                 d_loss_fake = criterion(outputs, fake_labels)

                 d_loss = d_loss_real + d_loss_fake
                 optimizer_d.zero_grad()
                 d_loss.backward()
                 optimizer_d.step()

                 # Train generator
                 outputs = discriminator(fake_data)
                 g_loss = criterion(outputs, real_labels)

                 optimizer_g.zero_grad()
                 g_loss.backward()
                 optimizer_g.step()

             print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}')
     ```

2. **Paper: "Auto-Encoding Variational Bayes" (Kingma, Welling, 2013)**
   - **Summary:**
     - This paper introduces Variational Autoencoders (VAEs), which combine principles from variational inference and autoencoders.
   - **Key Contributions:**
     - Developed a probabilistic approach to autoencoding, allowing for the generation of new data points.
   - **Implementation:**
     ```python
     import torch
     import torch.nn as nn
     import torch.optim as optim

     class VAE(nn.Module):
         def __init__(self):
             super(VAE, self).__init__()
             self.fc1 = nn.Linear(784, 400)
             self.fc21 = nn.Linear(400, 20)
             self.fc22 = nn.Linear(400, 20)
             self.fc3 = nn.Linear(20, 400)
             self.fc4 = nn.Linear(400, 784)

         def encode(self, x):
             h1 = torch.relu(self.fc1(x))
             return self.fc21(h1), self.fc22(h1)

         def reparameterize(self, mu, logvar):
             std = torch.exp(0.5*logvar)
             eps = torch.randn_like(std)
             return mu + eps*std

         def decode(self, z):
             h3 = torch.relu(self.fc3(z))
             return torch.sigmoid(self.fc4(h3))

         def forward(self, x):
             mu, logvar = self.encode(x.view(-1, 784))
             z = self.reparameterize(mu, logvar)
             return self.decode(z), mu, logvar

     def loss_function(recon_x, x, mu, logvar):
         BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
         KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
         return BCE + KLD

     def train_vae(model, data_loader, num_epochs, lr):
         optimizer = optim.Adam(model.parameters(), lr=lr)
         model.train()

         for epoch in range(num_epochs):
             train_loss = 0
             for data, _ in data_loader:
                 optimizer.zero_grad()
                 recon_batch, mu, logvar = model(data)
                 loss = loss_function(recon_batch, data, mu, logvar)
                 loss.backward()
                 train_loss += loss.item()
                 optimizer.step()

             print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(data_loader.dataset)}')
     ```

