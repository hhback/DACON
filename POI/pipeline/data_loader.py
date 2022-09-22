import numpy as np
import tensorflow as tf
import cv2

class DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(
        self,
        img_path_list,
        img_size,
        sentence,
        max_length,
        tokenizer,
        labels,
        batch_size,
        seed,
        shuffle=True,
        include_targets=True,
    ):
        self.img_path_list = img_path_list
        self.img_size = img_size
        self.sentence = sentence
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.labels = labels
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.include_targets = include_targets
        self.indexes = np.arange(len(self.sentence))
        
        self.on_epoch_end()
        
    def preprocess_image(self, path):

        image = cv2.imread(path)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = image / 255
        image = image.reshape(-1, self.img_size, self.img_size, 3)[:, :, :, [2, 1, 0]]

        return image

    def __len__(self):

        return len(self.sentence) // self.batch_size

    def __getitem__(self, idx):

        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        # Image
        img_path = self.img_path_list[indexes]
        image = np.concatenate([self.preprocess_image(img) for img in img_path])

        # Text
        sentence = self.sentence[indexes]

        encoded = self.tokenizer.batch_encode_plus(
            sentence.tolist(),
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="tf",
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")

            return [image, input_ids, attention_masks, token_type_ids], labels
        else:
            return [image, input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):

        if self.shuffle:
            np.random.RandomState().shuffle(self.indexes)