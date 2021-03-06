import tensorflow as tf
import imageio
import numpy as np
import imgaug


class Converter:
    def __init__(self, weights_path):
        self.keras_model = tf.keras.models.load_model(weights_path)
        self.keras_model.summary()

    def contvert_to_tflite(self, output_tflite_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        converter.experimental_new_converter = True

        tflite_model = converter.convert()
        open(output_tflite_path, "wb").write(tflite_model)

    def convert_h5_to_tflite(self, h5_file, output_tflite_path):
        model = tf.keras.models.load_model(h5_file)
        converter = tf.lite.TFLiteConverter.from_session(sess=tf.keras.backend.get_session(),
                                                         input_tensors=model.inputs,
                                                         output_tensors=model.outputs)
        tflite_model = converter.convert()
        open(output_tflite_path, "wb").write(tflite_model)

    def contvert_to_tflite_from_keras(self, output_tflite_path):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.keras_model)
        tflite_model = converter.convert()
        open(output_tflite_path, "wb").write(tflite_model)

    def test_segmentation_prediction(self, i_file: str, o_file: str):
        self.ta = self.cfg.transformAugmentor()
        i_content = imageio.imread(o_file)
        ai = self.ta.augment_image(i_content)
        data = np.array([ai])
        res = self.keras_model.predict(data)

        map = imgaug.SegmentationMapOnImage(res[0], res[0].shape)
        scaledMap = imgaug.augmenters.Scale(
            {"height": i_content.shape[0], "width": i_content.shape[1]}).augment_segmentation_maps([map])
        imageio.imwrite(i_file,
                        imgaug.HeatmapsOnImage(scaledMap[0].arr, scaledMap[0].arr.shape).draw_on_image(i_content)[0])
