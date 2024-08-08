from coco import load_coco

coco_inst = load_coco()
coco_inst.save_embedded_caption_and_img_descriptors()
print("retrieving data to make sure the shapes are okay")
caption_embeddings, img_descriptors = coco_inst.get_embedded_caption_and_img_descriptors()

print(caption_embeddings.shape)
print(img_descriptors.shape)