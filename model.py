from keras.layers import Embedding
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Flatten, Conv1D, SpatialDropout1D, MaxPooling1D,AveragePooling1D, Bidirectional, merge, concatenate, Input, Dropout, LSTM

def build_model(embedding_size, max_words, y_dim, num_filters, filter_sizes, dropout, vocab_size, embed=False, embedding_matrix=[], embedding_train=False, pool_padding='valid'):
    embed_input = Input(shape=(max_words,))
    if embed == True:
        x = Embedding(vocab_size, embedding_size,weights=[embedding_matrix], input_length=max_words, trainable=embedding_train)(embed_input)
    else:
        x = Embedding(vocab_size, embedding_size, input_length=max_words)(embed_input)
    pooled_outputs = []
    for i in range(len(filter_sizes)):
        conv = Conv1D(num_filters, kernel_size=filter_sizes[i], padding='valid', activation='relu')(x)
        conv = MaxPooling1D(pool_size=max_words-filter_sizes[i]+1)(conv)           
        pooled_outputs.append(conv)
    merge = concatenate(pooled_outputs)
    
    x = Dense(30, activation='relu')(merge)
    x = Dropout(dropout)(x)
    x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.5, recurrent_dropout=0.1))(x)
    x = Dense(30, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(y_dim, activation='softmax')(x)

    model = Model(inputs=embed_input,outputs=x)

#     model.compile(optimizer='adam',loss = 'categorical_crossentropy', metrics = ['acc'])
#     print(model.summary())
    
#     from keras.utils import plot_model
#     plot_model(model, to_file='shared_input_layer.png')
    
    return model
