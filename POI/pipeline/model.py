import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

def image_model(image_pretrained_model, image_size):
    
    input_image = layers.Input(
        shape=(image_size, image_size, 3), dtype=tf.float32, name="image"
    )

    backbone_cnn = image_pretrained_model(input_image)
    pooling = layers.GlobalAveragePooling2D()(backbone_cnn)
    output = layers.Dense(1024)(pooling)
    
    model_cnn = tf.keras.models.Model(inputs=[input_image], outputs=output)

    for layer in model_cnn.layers:
        layer.trainable = True
        
    return model_cnn

def text_model(pretrained_bert, max_length):

    input_ids = layers.Input(
    shape=(max_length,), dtype=tf.int32, name="input_ids"
    )

    attention_masks = layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )

    token_type_ids = layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )

    bert_model = pretrained_bert

    # bert_model.trainable = False

    bert_output = bert_model(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    x = bert_output.last_hidden_state
    pooling = tf.keras.layers.GlobalAveragePooling1D()(x)

    model_lstm = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=pooling
    )
    
    for layer in model_lstm.layers:
        layer.trainable = True
        
    return model_lstm

def mutlitmodal_model(image, text, image_size, max_length, num_class):
    
    image_input = layers.Input(shape = (image_size, image_size, 3), dtype=tf.float32,
                           name = "image")
    input_word_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                           name="input_word_ids")
    input_mask = layers.Input(shape=(max_length,), dtype=tf.int32,
                                       name="input_mask")
    segment_ids = layers.Input(shape=(max_length,), dtype=tf.int32,
                                        name="segment_ids")

    image_side = image(image_input, image_size)
    text_side = text([input_word_ids, input_mask, segment_ids])

    # Concatenate features from images and texts
    merged = layers.Concatenate()([image_side, text_side])
    #merged = layers.Dense(256, activation = 'relu')(merged)
    output = layers.Dense(num_class, activation='softmax', name = "class")(merged)
    
    model_multi = models.Model([
                      image_input,
                      input_word_ids,
                      input_mask,
                      segment_ids], output)
    
    return model_multi