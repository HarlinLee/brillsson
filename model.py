import numpy as np
import tensorflow as tf
import larq as lq
import larq_zoo as lqz

class Spectrogram(tf.keras.layers.Layer):
  def __init__(self):
    super(Spectrogram, self).__init__()
    self._augment = augment
    self._sample_rate = 16000
    self._frame_length = 400
    self._frame_step = 160
    self._fft_length = 1024
    self._n_mels = 64
    self._fmin = 40.0
    self._fmax = 7800.0

  def call(self, waveform):
    stfts = tf.signal.stft(waveform, frame_length=self._frame_length, 
              frame_step=self._frame_step,
              fft_length=self._fft_length)
    spectrograms = tf.abs(stfts)

    num_spectrogram_bins = stfts.shape[-1] 
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = self._fmin, self._fmax, self._n_mels
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, self._sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
      spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
      linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    return log_mel_spectrograms[...,tf.newaxis]

class BinaryModel:
  def create_model(self, input_shape, num_classes, 
    model_type, hidden_size=None, add_classifier=False, 
    add_extra_layer=True, n_mels=64):

    if add_extra_layer:
      assert hidden_size is not None and hidden_size > 0

    if add_classifier:
      assert num_classes is not None and num_classes > 0

    encoder_inputs = tf.keras.layers.Input((None, n_mels, 1))
    if model_type == "quicknet":
      model = lqz.sota.QuickNet(
        input_tensor=encoder_inputs,
        weights=None, include_top=False)
    elif model_type == "meliusnet":
      model = lqz.literature.MeliusNet22(
        input_tensor=encoder_inputs,
        weights=None, include_top=False)
    elif model_type == "densenet":
      model = lqz.literature.BinaryDenseNet28(
        input_tensor=encoder_inputs, 
        weights=None, include_top=False)      
    encoder_outputs = model(encoder_inputs)
    encoder_outputs = tf.keras.layers.GlobalMaxPool2D()(
      encoder_outputs)
    encoder = tf.keras.Model(
      inputs=encoder_inputs, 
      outputs=encoder_outputs)

    inputs = tf.keras.layers.Input(input_shape)
    x = Spectrogram()(inputs)
    x = encoder(x)
    if add_extra_layer:
      x = lq.layers.QuantDense(hidden_size, activation=None)(x)
    if add_classifier:
      x = lq.layers.QuantDense(num_classes, activation=None)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

  def create_classification_model(self, input_shape, 
    encoder, num_classes):
    inputs = tf.keras.layers.Input(input_shape)
    x = Spectrogram()(inputs)
    x = encoder(x)
    outputs = lq.layers.QuantDense(num_classes, activation=None)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class Distiller(tf.keras.Model):
  def __init__(self, student, teacher):
    super(Distiller, self).__init__()
    self.teacher = teacher
    self.student = student

  def compile(
    self,
    optimizer,
    metrics,
    distillation_loss_fn,
    temperature=2,
  ):
    super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)

    self.distillation_loss_fn = distillation_loss_fn
    self.temperature = temperature

  def train_step(self, data):
    x = data

    # Forward pass of teacher
    teacher_predictions = self.teacher(x, training=False)

    with tf.GradientTape() as tape:
      # Forward pass of student
      student_predictions = self.student(x, training=True)

      distillation_loss = self.distillation_loss_fn(
        y_true=teacher_predictions,
        y_pred=student_predictions
      )

      loss = distillation_loss

    # Compute gradients
    trainable_vars = self.student.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # Update weights
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Update the metrics configured in `compile()`.
    self.compiled_metrics.update_state(teacher_predictions, student_predictions)

    # Return a dict of performance
    results = {m.name: m.result() for m in self.metrics}
    results.update(
        {"distillation_loss": distillation_loss}
    )
    return results