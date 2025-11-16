Plan: Recreate the famous CLIP model to some degree (obviously I don't have matching compute capabilities).

According the research paper (https://arxiv.org/pdf/2103.00020):

Used a text encoder: Text Transformer.
Used a vision encoder: ResNet50 with He init, antialiased rect-2 blur pooling, replaced global average pooling with an attention pooling which is implemented as a multihead QKV single attention layer.
Techniques: Adam Optimizer, decoupled weight decay regularization to all non gain or bias weights, cosine learning rate decay. 

Hyperparams found through grid search random search, and manual tuning when trained for one epoch before being adjusted for a scaled up model. Clipped to prevent scaling logits, learnable temperature parameter gamma was init to 0.07, minibatch size of 32,768.

Trained in batches of images and captions matched against each other where model attempts to calculate highest cosine similarity between matching pairs and minimizing amongst the lowest pairs.

Plan:
Note: Visual and text encoder need to project to the same dim (256, 512, or 768)
Visual Encoders: ResNet50 (with changes implemented from before and ImageNet weights), ViT-Tiny/Small (5M, 22M params)
Text Encoders: MiniLM(22-33M), DistilBERT(66M)
Dataset: LAION subsets (100k, 1M, 10M)
Hyperparams: temperature, learning rate, cosine schedule, weight decay, AdamW optimizer
Batch Size: Need to be large
Precision: bfloat16
Init: Kaiming, He