# TensorFlow is an open source machine learning library
import tensorflow as tf
# Keras is TensorFlow's high-level API for deep learning
from tensorflow import keras

import json

def save_model(model, name=None, folder="storedANN"):
    """
        Save as model.h5, model_weights.h5, and model.json

        from https://jiafulow.github.io/blog/2021/02/17/simple-fully-connected-nn-firmware-using-hls4ml/
    """

    if name is None:
        name = model.name

    print("Saving '" + name +"'")
    print("Saving model:")
    try:
        model.save(folder + "/" + name + '.h5')
    except Exception as e:
        print(e)
        print("Trying second approach:")
        model.save(folder + "/" + name )

    print("Saving model weights:")
    try:
        model.save_weights(folder + "/" + name + '_weights.h5')
    except Exception as e:
        print(e)

    print("Saving model JSON:")
    try:
        with open(folder + "/" + name + '.json', 'w') as outfile:
            json.dump(json.loads(model.to_json()), outfile, indent=4, sort_keys=False)
    except Exception as e:
        print(e)

    return

def save_model_ONNX(model, name=None, folder="storedANN"):
    # import ONNX for exporting the keras model
    import onnx
    import tf2onnx

    # based on https://medium.com/nerd-for-tech/how-to-convert-tensorflow2-model-to-onnx-using-tf2onnx-when-there-is-custom-ops-6e703376ef20
    # 1. Load the Tensorflow Model <- skipped, because we have already TF model loaded

    # 2. Convert the Model to Concrete Function
    full_model = tf.function(lambda inputs: model(inputs))
    full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])

    # 2.1 Persist the Input and Output Parameters
    input_names = [inp.name for inp in full_model.inputs]
    output_names = [out.name for out in full_model.outputs]
    print("Inputs:", input_names)
    print("Outputs:", output_names)

    # 3. Freeze the Model
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    # 4. Convert the Model in Single Step by using Extra Opset [Critical Code]
    from tf2onnx import tf_loader
    from tf2onnx.tfonnx import process_tf_graph
    from tf2onnx.optimizer import optimize_graph
    from tf2onnx import utils, constants
    from tf2onnx.handler import tf_op

    extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]
    with tf.Graph().as_default() as tf_graph:
        tf.import_graph_def(frozen_func.graph.as_graph_def(), name='')

    with tf_loader.tf_session(graph=tf_graph):
        g = process_tf_graph(tf_graph, input_names=input_names, output_names=output_names, extra_opset=extra_opset)

    onnx_graph = optimize_graph(g)
    model_proto = onnx_graph.make_model(model.name+"conv")
    utils.save_protobuf(folder+"/"+model.name+".onnx", model_proto)
    print("Conversion complete!")

    return

def save_model_image(model):
    # save an image of the ANN
    tf.keras.utils.plot_model(model, 
            to_file=model.name+".png",    # output file name
            #show_layer_activations=True,  # show activation functions
            show_layer_names=True,        # show layer names
            show_dtype=True,              # show datatype
            show_shapes=True,             # show input / output shapes
            rankdir='LR'                  # left to right image
        )

def load_quant_model(name, folder="storedANN"):
    from keras.models import model_from_json
    from qkeras.utils import _add_supported_quantized_objects
    co = {}
    _add_supported_quantized_objects(co)

    # load json and create model
    with open(folder + "/" + name + '.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json, custom_objects=co)
    # load weights into new model
    model.load_weights(folder + "/" + name + '_weights.h5')
    print("Loaded "+ model.name +" from disk.")

    return model

def save_quant_model(model, name=None, folder="storedANN"):
    """
        Save as model.h5, model_weights.h5, and model.json

        from https://jiafulow.github.io/blog/2021/02/17/simple-fully-connected-nn-firmware-using-hls4ml/
    """


    if name is None:
        name = model.name

    print("Saving '" + name +"'")
    print("Saving model:")
    try:
        model.save(folder + "/" + name + '.h5')
    except Exception as e:
        print(e)
        print("Trying second approach:")
        model.save(folder + "/" + name )

    print("Saving model weights:")
    try:
        model_save_quantized_weights(model, filename=folder + "/" + name + '_weights.h5')
    except Exception as e:
        print(e)

    print("Saving model JSON:")
    try:
        with open(folder + "/" + name + '.json', 'w') as outfile:
            json.dump(json.loads(model.to_json()), outfile, indent=4, sort_keys=False)
    except Exception as e:
        print(e)

    return
