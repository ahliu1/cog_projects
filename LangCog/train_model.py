import numpy as np
from mynn.optimizers.sgd import SGD
from mygrad.nnet.losses.margin_ranking_loss import margin_ranking_loss
import mygrad as mg
from noggin import create_plot
from model_creation import Model
from coco import load_coco
from get_data import get_training_and_validation_data
import pickle
# plotter, fig, ax = create_plot(["loss"])

coco = load_coco()
model = Model()  # D_full=512, D_hidden=200
optimizer = SGD(model.parameters, learning_rate=1e-3, momentum=0.9)

caption_descriptors, true_image_descriptors = coco.get_embedded_caption_and_img_descriptors()
batch_size = 32
num_epochs = 1

true_image_descriptors_train = true_image_descriptors[:int(true_image_descriptors.size*0.8)] # get 4/5 of data
caption_descriptors_train = caption_descriptors[:int(caption_descriptors.size*0.8)]
confuser_img_descriptors_train = np.copy(true_image_descriptors_train)

true_image_descriptors_valid = true_image_descriptors[int(true_image_descriptors.size*0.8):] # get 1/5 of data
caption_descriptors_valid = caption_descriptors[int(caption_descriptors.size*0.8):]
confuser_img_descriptors_valid = np.copy(true_image_descriptors_valid)
idxs_train = np.arange(len(caption_descriptors_train))
idxs_valid = np.arange(len(caption_descriptors_valid))

for epoch_cnt in range(num_epochs):
    np.random.shuffle(idxs_train)
    np.random.shuffle(confuser_img_descriptors_train)
    for batch_cnt in range(0, len(caption_descriptors_train) // batch_size):
        batch_indices = idxs_train[batch_cnt * batch_size: (batch_cnt + 1) * batch_size]
        embedded_caption_batch = caption_descriptors_train[batch_indices]
        true_batch = true_image_descriptors_train[batch_indices]
        confuser_batch = confuser_img_descriptors_train[batch_indices]
        
        true_embed = model(true_batch)
        confuse_embed = model(confuser_batch)
        
        embedded_caption_batch_norm = embedded_caption_batch / np.linalg.norm(embedded_caption_batch, axis=1).reshape(-1, 1)
        sim_true = mg.einsum("nd,nd -> n", true_embed, embedded_caption_batch_norm)
        sim_confusor = mg.einsum("nd,nd -> n", confuse_embed, embedded_caption_batch_norm)

        loss = margin_ranking_loss(sim_true, sim_confusor,1, margin=0.25)
        if batch_cnt % 50 == 0:
            print("train loss: ", loss)
        loss.backward()
        optimizer.step()

        # plotter.set_train_batch({"loss": loss.item()}, batch_size=batch_size)
    
# Plot loss of validation data
np.random.shuffle(idxs_valid)
np.random.shuffle(confuser_img_descriptors_valid)
for batch_cnt in range(0, len(caption_descriptors_valid) // batch_size):
    batch_indices = idxs_valid[batch_cnt * batch_size: (batch_cnt + 1) * batch_size]
    embedded_caption_batch = caption_descriptors_valid[batch_indices]
    true_batch = true_image_descriptors_valid[batch_indices]
    confuser_batch = confuser_img_descriptors_valid[batch_indices]
    
    true_embed = model(true_batch)
    confuse_embed = model(confuser_batch)
    

    sim_true = mg.einsum("nd,nd -> n", true_embed, embedded_caption_batch)
    sim_confusor = mg.einsum("nd,nd -> n", confuse_embed, embedded_caption_batch)

    # margin ranking loss
    loss = margin_ranking_loss(sim_true, sim_confusor,1, margin=0.25)
    if batch_cnt % 50 == 0:
        print("valid loss: ",loss)
    
# plotter.plot()
# SAVE DATA
with open("datasets/model_parameters.pkl","w+b") as f:
    pickle.dump(model.parameters,f)
