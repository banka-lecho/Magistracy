import tensorflow as tf
import numpy as np
import random
import time
import math
import contextlib
import os
import hashlib

from ArithmeticCoder import ArithmeticEncoder, ArithmeticDecoder, BitOutputStream, BitInputStream

os.environ['TF_DETERMINISTIC_OPS'] = '1'

# The batch size for training
BATCH_SIZE = 256
# The sequence length for training
SEQ_LENGTH = 4
# The number of units in each LSTM layer
RNN_UNITS = 256
# The number of GRU layers
NUM_LAYERS = 1
# The size of the embedding layer
EMBEDDING_SIZE = 512
# The initial learning rate for optimizer
START_LR = 0.005
# The final learning rate for optimizer
END_LR = 0.001
# The mode for the program, "compress", "decompress", "both"
mode = 'both'

PATH_TO_FILE = "data/enwik5"
PATH_TO_COMPRESSED = PATH_TO_FILE + "_compressed_my.dat"
PATH_TO_DECOMPRESSED = PATH_TO_FILE + "_decompressed_my.dat"


def build_model(vocab_size):
    inputs = [
        tf.keras.Input(shape=[SEQ_LENGTH, ], batch_size=BATCH_SIZE),
        tf.keras.Input(shape=(RNN_UNITS,))
    ]

    embedding = tf.keras.layers.Embedding(
        vocab_size, EMBEDDING_SIZE)(inputs[0])

    predictions, state_h = tf.keras.layers.GRU(
        RNN_UNITS, return_sequences=True, return_state=True,
        recurrent_initializer='glorot_uniform',
        reset_after=True)(embedding, initial_state=inputs[1])

    last_timestep = tf.keras.layers.Lambda(
        lambda x: x[:, SEQ_LENGTH - 1, :])(predictions)

    dense = tf.keras.layers.Dense(vocab_size, name='dense_logits')(last_timestep)
    temperature = 0.8
    output = tf.keras.layers.Lambda(lambda x: tf.keras.activations.softmax(x / temperature),
                                    name='predictions')(dense)

    model = tf.keras.Model(inputs=inputs, outputs=[output, state_h])
    return model


def get_symbol(index, length, freq, coder, compress, data):
    """Runs arithmetic coding and returns the next symbol.

    Args:
        index: Int, position of the symbol in the file.
        length: Int, size limit of the file.
        freq: ndarray, predicted symbol probabilities.
        coder: this is the arithmetic coder.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.

    Returns:
        The next symbol, or 0 if "index" is over the file size limit.
    """
    symbol = 0
    if index < length:
        if compress:
            symbol = data[index]
            coder.write(freq, symbol)
        else:
            symbol = coder.read(freq)
            data[index] = symbol
    return symbol


def train(pos, seq_input, length, vocab_size, coder, model, optimizer, compress,
          data, states):
    """Runs one training step.

    Args:
        pos: Int, position in the file for the current symbol for the *first* batch.
        seq_input: Tensor, containing the last SEQ_LENGTH inputs for the model.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        model: the model to generate predictions.
        optimizer: optimizer used to train the model.
        compress: Boolean, True if compressing, False if decompressing.
        data: List containing each symbol in the file.
        states: List containing state information for the layers of the model.

    Returns:
        seq_input: Tensor, containing the last SEQ_LENGTH inputs for the model.
        cross_entropy: cross entropy numerator.
        denom: cross entropy denominator.
    """
    loss = cross_entropy = denom = 0
    split = math.ceil(length / BATCH_SIZE)
    # Keep track of operations while running the forward pass for automatic
    # differentiation.
    with tf.GradientTape() as tape:
        # The model inputs contain both seq_input and the states for each layer.
        inputs = states.pop(0)
        inputs.insert(0, seq_input)
        predictions, new_state = model(inputs)
        states.append([new_state])
        p = predictions.numpy()
        symbols = []
        # When the last batch reaches the end of the file, we start giving it "0"
        # as input. We use a mask to prevent this from influencing the gradients.
        mask = []
        # Go over each batch to run the arithmetic coding and prepare the next
        # input.
        for i in range(BATCH_SIZE):
            # The "10000000" is used to convert floats into large integers (since
            # the arithmetic coder works on integers).
            freq = np.cumsum(p[i] * 10000000 + 1)
            index = pos + 1 + i * split
            symbol = get_symbol(index, length, freq, coder, compress, data)
            symbols.append(symbol)
            if index < length:
                prob = p[i][symbol]
                if prob <= 0:
                    # Set a small value to avoid error with log2.
                    prob = 0.000001
                cross_entropy += math.log2(prob)
                denom += 1
                mask.append(1.0)
            else:
                mask.append(0.0)
        # "input_one_hot" will be used both for the loss function and for the next
        # input.
        input_one_hot = tf.one_hot(symbols, vocab_size)
        loss = tf.keras.losses.categorical_crossentropy(
            input_one_hot, predictions, from_logits=False) * tf.expand_dims(
            tf.convert_to_tensor(mask), 1)
        # scaled_loss = optimizer.get_scaled_loss(loss)
        # Remove the oldest input and append the new one.
        seq_input = tf.slice(seq_input, [0, 1], [BATCH_SIZE, SEQ_LENGTH - 1])
        seq_input = tf.concat([seq_input, tf.expand_dims(symbols, 1)], 1)
    # Run the backwards pass to update model weights.
    gradients = tape.gradient(loss, model.trainable_variables)
    # grads = optimizer.get_unscaled_gradients(scaled_gradients)
    # Gradient clipping to make training more robust.
    capped_grads = [tf.clip_by_norm(grad, 4) for grad in gradients]
    optimizer.apply_gradients(zip(capped_grads, model.trainable_variables))
    return (seq_input, cross_entropy, denom)


def reset_seed():
    """Initializes various random seeds to help with determinism."""
    SEED = 1234
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def process(compress, length, vocab_size, coder, data):
    """This runs compression/decompression.

    Args:
        compress: Boolean, True if compressing, False if decompressing.
        length: Int, size limit of the file.
        vocab_size: Int, size of the vocabulary.
        coder: this is the arithmetic coder.
        data: List containing each symbol in the file.
    """
    start = time.time()
    reset_seed()
    model = build_model(vocab_size=vocab_size)
    model.summary()

    # Try to split the file into equal size pieces for the different batches. The
    # last batch may have fewer characters if the file can't be split equally.
    split = math.ceil(length / BATCH_SIZE)

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        START_LR,
        split,
        END_LR,
        power=1.0)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_fn, beta_1=0, beta_2=0.9999, epsilon=1e-5)

    # Use a uniform distribution for predicting the first batch of symbols. The
    # "10000000" is used to convert floats into large integers (since the
    # arithmetic coder works on integers).
    freq = np.cumsum(np.full(vocab_size, (1.0 / vocab_size)) * 10000000 + 1)
    # Construct the first set of input characters for training.
    symbols = []
    for i in range(BATCH_SIZE):
        symbols.append(get_symbol(i * split, length, freq, coder, compress, data))
    # Replicate the input tensor SEQ_LENGTH times, to match the input format.
    seq_input = tf.tile(tf.expand_dims(symbols, 1), [1, SEQ_LENGTH])
    pos = cross_entropy = denom = 0
    template = '{:0.2f}%\tcross entropy: {:0.2f}\ttime: {:0.2f}'
    # This will keep track of layer states. Initialize them to zeros.
    states = []
    for i in range(SEQ_LENGTH):
        states.append([tf.zeros([BATCH_SIZE, RNN_UNITS])])
    # Keep repeating the training step until we get to the end of the file.
    while pos < split:
        seq_input, ce, d = train(pos, seq_input, length, vocab_size, coder, model,
                                 optimizer, compress, data, states)
        cross_entropy += ce
        denom += d
        pos += 1
        if pos % 20 == 0:
            percentage = 100 * pos / split
            if percentage >= 100:
                continue
            print(template.format(percentage, -cross_entropy / denom, time.time() - start))
    if compress:
        coder.finish()
    print(template.format(100, -cross_entropy / length, time.time() - start))
    print(f"BPC: {(-cross_entropy / length):.3f}")
    print(f"PPL: {2 ** (-cross_entropy / length):.3f}")


def compession():
    # int_list will contain the characters of the file.
    int_list = []
    text = open(PATH_TO_FILE, 'rb').read()
    vocab = sorted(set(text))
    vocab_size = len(vocab)
    # Creating a mapping from unique characters to indexes.
    char2idx = {u: i for i, u in enumerate(vocab)}
    for idx, c in enumerate(text):
        int_list.append(char2idx[c])

    # Round up to a multiple of 8 to improve performance.
    vocab_size = math.ceil(vocab_size / 8) * 8
    file_len = len(int_list)
    print('Length of file: {} symbols'.format(file_len))
    print('Vocabulary size: {}'.format(vocab_size))

    with open(PATH_TO_COMPRESSED, "wb") as out, contextlib.closing(BitOutputStream(out)) as bitout:
        length = len(int_list)
        # Write the original file length to the compressed file header.
        out.write(length.to_bytes(5, byteorder='big', signed=False))
        # Write 256 bits to the compressed file header to keep track of the vocabulary.
        for i in range(256):
            if i in char2idx:
                bitout.write(1)
            else:
                bitout.write(0)
        enc = ArithmeticEncoder(32, bitout)
        process(True, length, vocab_size, enc, int_list)


def decompression():
    with open(PATH_TO_COMPRESSED, "rb") as inp, open(PATH_TO_DECOMPRESSED, "wb") as out:
        # Read the original file size from the header.
        length = int.from_bytes(inp.read()[:5], byteorder='big')
        inp.seek(5)
        # Create a list to store the file characters.
        output = [0] * length
        bitin = BitInputStream(inp)

        # Get the vocabulary from the file header.
        vocab = []
        for i in range(256):
            if bitin.read():
                vocab.append(i)
        vocab_size = len(vocab)
        # Round up to a multiple of 8 to improve performance.
        vocab_size = math.ceil(vocab_size / 8) * 8
        dec = ArithmeticDecoder(32, bitin)
        process(False, length, vocab_size, dec, output)
        # The decompressed data is stored in the "output" list. We can now write the
        # data to file (based on the type of preprocessing used).

        # Convert indexes back to the original characters.
        idx2char = np.array(vocab)
        for i in range(length):
            out.write(bytes((idx2char[output[i]],)))


def main():
    start = time.time()
    if mode == 'compress' or mode == 'both':
        compession()
        print(f"Original size: {os.path.getsize(PATH_TO_FILE)} bytes")
        print(f"Compressed size: {os.path.getsize(PATH_TO_COMPRESSED)} bytes")
        print("Compression ratio:", os.path.getsize(PATH_TO_FILE) / os.path.getsize(PATH_TO_COMPRESSED))
    if mode == 'decompress' or mode == 'both':
        decompression()
        hash_dec = hashlib.md5(open(PATH_TO_DECOMPRESSED, 'rb').read()).hexdigest()
        hash_orig = hashlib.md5(open(PATH_TO_FILE, 'rb').read()).hexdigest()
        assert hash_dec == hash_orig
    print("Time spent: ", time.time() - start)


if __name__ == '__main__':
    main()
