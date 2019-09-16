# -*- coding:utf8 -*-



import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_io,graph_util
from tensorflow.python.platform import gfile
import coremltools
import keras

from model import build

import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Inference mult-attributes classifier')
    parser.add_argument('--model', type=str, 
        default=None, help='complete trained .h5 file')
    parser.add_argument('--save_dir', type=str, 
        default="./model", help='directory for saving exported models.')
    parser.add_argument('--gpu', type=str, default='0',
        help='the gpu id to train net')
    parser.add_argument('--feature_aggregate', type=bool, default=False,
        help='whether do feature aggregation.')
    parser.add_argument('--image_size', type=int, default=224,
        help='input image size.')
    parser.add_argument('--base_model', type=str, default="mobilenetv2",
        choices=["MobileNetV2","DenseNet121","Xception","VGG16","VGG19"], help='select a backbone model')
    return parser.parse_args()



def export_pb(session, input_names, output_names, model_dir, model_name):
    model_name = os.path.basename(model_name).replace(".h5", "")
    graph_def = session.graph_def
    for node in graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
            # input0: ref: Should be from a Variable node. May be uninitialized.
            # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]
    # generate protobuf
    constant_graph = graph_util.convert_variables_to_constants(session, graph_def, output_names)
    transformed_graph_def = TransformGraph(constant_graph,
                                   input_names,
                                   output_names,
                                   ["add_default_attributes",
                                    "remove_nodes(op=Identity,op=CheckNumerics)",
                                    "fold_constants(ignore_errors=true)",
                                    "fold_batch_norms",])
    tf.train.write_graph(transformed_graph_def, model_dir, '%s.pb' % model_name, as_text=False)
    
    
    
def export_coreml(model, input_names, output_names, model_dir, model_name):
    model_name = os.path.basename(model_name).replace(".h5", "")
    coreml_model = coremltools.converters.keras.convert(model, 
                                                        input_names=input_names,
                                                        image_input_names=input_names,
                                                        output_names=output_names)
    # get the spec from the model
    spec = coreml_model.get_spec()
    # create a local reference to the Float32 type
    Float32 = coremltools.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    # set the output shape for the segmentation to Float32
    spec.description.output[0].type.multiArrayType.dataType = Float32
    spec.description.output[1].type.multiArrayType.dataType = Float32
    spec.description.output[2].type.multiArrayType.dataType = Float32

    coremltools.utils.save_spec(spec, os.path.join(model_dir, "%s.mlmodel" % model_name))


def export_tflite(sess, inputs, outputs, model_dir, model_name):
    model_name = os.path.basename(model_name).replace(".h5", "")
    converter = tf.lite.TFLiteConverter.from_session(sess, inputs, outputs)
    output_model = converter.convert()
    with open(os.path.join(model_dir, "%s.tflite" % model_name), "wb") as fo:
        fo.write(output_model)

if __name__ == "__main__":
    ## parse args
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    with open("config.json") as fi:
        config=json.load(fi)
    ## define model
    features = config.get(args.base_model) if args.feature_aggregate else None
    model = build(input_shape=(args.image_size,args.image_size,3), 
                  base_model=args.base_model,
                  feature_layers=features, 
                  train_backbone=False,
                  weight_decay=0)
    model.load_weights(args.model)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = keras.backend.get_session()
    ## export pb file
    export_pb(sess, [i.op.name for i in model.inputs], [i.op.name for i in model.outputs], args.save_dir, args.model)
    ## export coreml
    export_coreml(model, model.input.op.name, [i.op.name for i in model.outputs], args.save_dir, args.model)
    ## export tflite
    export_tflite(sess, model.inputs, model.outputs, args.save_dir, args.model)
    print([i.op.name for i in model.inputs])
    print([i.op.name for i in model.outputs])