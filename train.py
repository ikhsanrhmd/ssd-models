#coding=utf-8
import os
import sys
import stat
import math
import shutil
import platform
if platform.system() == "Windows":
    caffe_root = "D:/CNN/ssd"
else:
    caffe_root = os.path.expanduser('~') + "/CNN/ssd"
sys.path.insert(0, caffe_root + '/python')
import caffe
from caffe.model_libs import *
from caffe import params as P
from mrdet.model import build_model
#for detatil, please refer to https://blog.csdn.net/chenlufei_i/article/details/80068953?utm_medium=distribute.wap_relevant.none-task-blog-title-2
datasetname = "insect"
arch = 'aizoo28'
if len(sys.argv) > 1:
  datasetname = sys.argv[1]
if len(sys.argv) > 2:
  arch = sys.argv[2]
resize_width = 320
resize_height = 320
if arch == 'svd15':
  resize_width = 90
  resize_height = 160
datasets={
    "voc":{"num_classes":21,"num_test_image":4952,"train":"trainval","val":"test"},
    "Face":{"num_classes":2,"num_test_image":3200,"train":"train","val":"val"},
    "fddb":{"num_classes":2,"num_test_image":285,"train":"trainval","val":"test"},
    "Car":{"num_classes":2,"num_test_image":10000,"train":"train","val":"val"},
    "Head":{"num_classes":2,"num_test_image":493,"train":"train","val":"val"},
    "Person":{"num_classes":2,"num_test_image":994,"train":"train","val":"val"},
    "Hand":{"num_classes":2,"num_test_image":13024,"train":"train","val":"val"},
    "Shoes":{"num_classes":2,"num_test_image":770,"train":"train","val":"val"},
    "tower":{"num_classes":2,"num_test_image":204,"train":"train","val":"val"},
    "Mask":{"num_classes":3,"num_test_image":1839,"train":"train","val":"val"},
    "insect":{"num_classes":7,"num_test_image":24,"train":"train","val":"val"}
}
resume_training=True
remove_old_models=True
label_map_file = datasetname+"/labelmap.prototxt"
num_classes = datasets[datasetname]['num_classes']
num_test_image = datasets[datasetname]['num_test_image']
batch_size = 16
resize = "{}x{}".format(resize_width, resize_height)
batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]
train_transform_param = {
        'mirror': True,
        'mean_value': [127.5,127.5,127.5],
        'scale': 0.007843,
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'distort_param': {
                'brightness_prob': 0.5,
                'brightness_delta': 32,
                'contrast_prob': 0.5,
                'contrast_lower': 0.5,
                'contrast_upper': 1.5,
                'hue_prob': 0.5,
                'hue_delta': 18,
                'saturation_prob': 0.5,
                'saturation_lower': 0.5,
                'saturation_upper': 1.5,
                'random_order_prob': 0.0,
                },
        'expand_param': {
                'prob': 0.5,
                'max_expand_ratio': 4.0,
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }
test_transform_param = {
        'mean_value': [127.5,127.5,127.5],
        'scale': 0.007843,
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [P.Resize.LINEAR],
                },
        }
use_batchnorm = True
lr_mult = 1
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.0004
else:
    # A learning rate for batch_size2 = 1, num_gpus = 1.
    base_lr = 0.00004
job_name = arch+"_{}".format(resize)
model_name = datasetname+"_{}".format(job_name)
save_dir = "output/"+datasetname+"_{}".format(job_name)   
# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
snapshot_dir = "output/"+datasetname+"_{}".format(job_name)
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
job_dir = "output/"+datasetname+"_{}".format(job_name)
job_file = "{}/{}.sh".format(job_dir, model_name)
name_size_file = "models/test_name_size.txt"
output_result_dir = "output/"+datasetname+"_"+"/result"
# MultiBoxLoss parameters.

share_location = True
background_label_id=0
train_on_diff_gt = True
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
ignore_cross_boundary_bbox = False
mining_type = P.MultiBoxLoss.MAX_NEGATIVE
neg_pos_ratio = 3.
loc_weight = (neg_pos_ratio + 1.) / 4.
multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.5,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'mining_type': mining_type,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
    }
loss_param = {
    'normalization': normalization_mode,
    }

cfgs = {
    "svd15":{"mbox_source_layers":['conv3_2', 'conv6_2', 'conv7_2'],
        "min_ratio":20,"max_ratio":40,"min_size":120,"steps":[4,8,16],
        "aspect_ratios":[[],[],[2]],"normalizations":[20, -1, -1]
        },
    "aizoo28":{"mbox_source_layers":['conv2d_3','conv2d_4','conv2d_5','conv2d_6','conv2d_7'],
       "min_ratio":20,"max_ratio":90,"min_size":120,"steps":[8,16,32,64,128],
       "aspect_ratios":[[2],[2],[2],[2],[2]],"normalizations":[20, -1, -1, -1, -1]
       },
}
cfg = cfgs[arch]
# parameters for generating priors.
# minimum dimension of input image
min_dim = cfg['min_size']
mbox_source_layers = cfg['mbox_source_layers']
# in percent %
min_ratio = cfg['min_ratio']
max_ratio = cfg['max_ratio']
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
#max_sizes = []
for ratio in range(min_ratio, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
#   max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 10 / 100.] + min_sizes
#max_sizes = [min_dim * 20 / 100.] + max_sizes
#min_sizes = [12,24,36]
steps = cfg['steps']
aspect_ratios =cfg['aspect_ratios']
# L2 normalize conv4_3.
normalizations = cfg['normalizations']
# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
flip = True
clip = False
num_gpus = 1
accum_batch_size = 64
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU

if normalization_mode == P.Loss.NONE:
  base_lr /= batch_size_per_device
elif normalization_mode == P.Loss.VALID:
  base_lr *= 25. / loc_weight
elif normalization_mode == P.Loss.FULL:
  # Roughly there are 2000 prior bboxes per image.
  # TODO(weiliu89): Estimate the exact # of priors.
  base_lr *= 2000.
# Evaluate on whole test set.
test_batch_size = 8
# Ideally test_batch_size should be divisible by num_test_image,
# otherwise mAP will be slightly off the true value.
test_iter = int(math.ceil(float(num_test_image) / test_batch_size))
solver_param = {
    # Train parameters
    'base_lr': base_lr,
    'weight_decay': 0.0005,
    'lr_policy': "exp",
    'gamma': 0.9999,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 120000,
    'snapshot': 1000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 1000,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': True,
    }
# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
#     'save_output_param': {
#         'output_directory': output_result_dir,
#         'output_name_prefix': "comp4_det_test_",
#         'output_format': "VOC",
#         'label_map_file': label_map_file,
#         'name_size_file': name_size_file,
#         'num_test_image': num_test_image,
#         },
    'keep_top_k': 200,
    'confidence_threshold': 0.01,
    'code_type': code_type,
    }
# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    #'name_size_file': name_size_file,
    }

### Hopefully you don't need to change the following ###
# Check file.
train_data = "data/"+datasetname+"/lmdb/"+datasets[datasetname]['train']+"_lmdb"
test_data = "data/"+datasetname+"/lmdb/"+datasets[datasetname]['val']+"_lmdb"
check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(label_map_file)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

# Create train net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
        train=True, output_label=True, label_map_file=label_map_file,
        transform_param=train_transform_param, batch_sampler=batch_sampler)
net = build_model(arch)(net)
# Create the MultiBoxLossLayer.
mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, #max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)
mbox_layers.append(net.label)
name = "mbox_loss"
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
        loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

with open(train_net_file, 'w') as f:
    f.write('name: "{}_train"\n'.format(model_name))
    f.write(str(net.to_proto()))

# Create deploy net.
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
        train=False, output_label=True, label_map_file=label_map_file,
        transform_param=test_transform_param)
net = build_model(arch)(net)
mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, #max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)
conf_name = "mbox_conf"
if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
  reshape_name = "{}_reshape".format(conf_name)
  net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
  softmax_name = "{}_softmax".format(conf_name)
  net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
  flatten_name = "{}_flatten".format(conf_name)
  net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
  mbox_layers[1] = net[flatten_name]
elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
  sigmoid_name = "{}_sigmoid".format(conf_name)
  net[sigmoid_name] = L.Sigmoid(net[conf_name])
  mbox_layers[1] = net[sigmoid_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
    detection_output_param=det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
    detection_evaluate_param=det_eval_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))

with open(test_net_file, 'w') as f:
    #print('name: "{}_test"'.format(model_name), file=f)
    #print(net.to_proto(), file=f)
    f.write('name: "{}_test"\n'.format(model_name))
    f.write(str(net.to_proto()))

# Remove the first and last layer from test net.
deploy_net = net
with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    del net_param.layer[0]
    del net_param.layer[-1]
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    #print(net_param, file=f)
    f.write(str(net_param))
# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    #print(solver, file=f)
    f.write(str(solver))
max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = ''
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)
  else:
    for file in os.listdir(snapshot_dir):
      if file.endswith(".caffemodel"):
        basename = os.path.splitext(file)[0]
        iter = int(basename.split("{}_iter_".format(model_name))[1])
        if iter > max_iter:
          max_iter = iter
    if max_iter > 0:
        train_src_param = '--weights="{}_iter_{}.caffemodel" \\\n'.format(snapshot_prefix, max_iter)
        max_iter = 0

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
gpus = "0"
for i in range(num_gpus-1):
  gpus = gpus + ","+str(i)
# Create job file.
with open(job_file, 'w') as f:
  f.write('{}/build/tools/caffe train \\\n'.format(caffe_root))
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Copy the python script to job_dir.
py_file = os.path.abspath(__file__)
shutil.copy(py_file, job_dir)

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
import subprocess
subprocess.call(job_file, shell=True)