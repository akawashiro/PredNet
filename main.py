# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
from PIL import Image

import chainer
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.functions.loss.mean_squared_error import mean_squared_error
import net
import logging
from datetime import datetime

start_time = str(datetime.now()).replace(" ","")
logging.basicConfig(filename='main' + start_time + '.log',
                    format='%(asctime)s %(message)s', level=logging.DEBUG)


def print_and_log(s):
    print s
    logging.info(s)


parser = argparse.ArgumentParser(
    description='PredNet')
parser.add_argument('--images', '-i', default='',
                    help='Path to image list file')
parser.add_argument('--sequences', '-seq', default='',
                    help='Path to sequence list file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--root', '-r', default='.',
                    help='Root directory path of sequence and image files')
parser.add_argument('--initmodel', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--size', '-s', default='160,128',
                    help='Size of target images. width,height (pixels)')
parser.add_argument('--channels', '-c', default='3,48,96,192',
                    help='Number of channels on each layers')
parser.add_argument('--offset', '-o', default='0,0',
                    help='Center offset of clipping input image (pixels)')
parser.add_argument('--ext', '-e', default=100, type=int,
                    help='Extended prediction on test (frames)')
parser.add_argument('--bprop', default=10, type=int,
                    help='Back propagation length (frames)')
parser.add_argument('--save', default=10000, type=int,
                    help='Period of save model and state (frames)')
parser.add_argument('--period', default=1000000, type=int,
                    help='Period of training (frames)')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)
parser.add_argument('--improve', default=0, type=int,
                    help='Use the PredNet of improved version')

args = parser.parse_args()


if (not args.images) and (not args.sequences):
    print_and_log('Please specify images or sequences')
    exit()

args.size = args.size.split(',')
for i in range(len(args.size)):
    args.size[i] = int(args.size[i])
args.channels = args.channels.split(',')
for i in range(len(args.channels)):
    args.channels[i] = int(args.channels[i])
args.offset = args.offset.split(',')
for i in range(len(args.offset)):
    args.offset[i] = int(args.offset[i])

if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# Create Model
#(height of image, width of image, channels on each layer)
if args.improve == 0:
    prednet = net.PredNet(args.size[0], args.size[1], args.channels)
if args.improve == 1:
    prednet = net.PredNet_simple_improve(
        args.size[0], args.size[1], args.channels)

model = L.Classifier(prednet, lossfun=mean_squared_error)
model.compute_accuracy = False
optimizer = optimizers.Adam()
optimizer.setup(model)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    print_and_log('Running on a GPU')
else:
    print_and_log('Running on a CPU')

# Init/Resume
if args.initmodel:
    print_and_log('Load model from' + str(args.initmodel))
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print_and_log('Load optimizer state from' + str(args.resume))
    serializers.load_npz(args.resume, optimizer)

if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('images'):
    os.makedirs('images')


def load_list(path, root):
    tuples = []
    for line in open(path):
        pair = line.strip().split()
        tuples.append(os.path.join(root, pair[0]))
    return tuples


def read_image(path):
    image = np.asarray(Image.open(path)).transpose(2, 0, 1)
    top = args.offset[1] + (image.shape[1] - args.size[1]) / 2
    left = args.offset[0] + (image.shape[2] - args.size[0]) / 2
    bottom = args.size[1] + top
    right = args.size[0] + left
    image = image[:, top:bottom, left:right].astype(np.float32)
    image /= 255
    return image


def write_image(image, path):
    image *= 255
    image = image.transpose(1, 2, 0)
    image = image.astype(np.uint8)
    result = Image.fromarray(image)
    result.save(path)

if args.images:
    sequencelist = [args.images]
else:
    sequencelist = load_list(args.sequences, args.root)

if args.test is True:
    for seq in range(len(sequencelist)):
        imagelist = load_list(sequencelist[seq], args.root)
        prednet.reset_state()
        loss = 0
        batchSize = 1
        x_batch = np.ndarray((batchSize, args.channels[0], args.size[
                             1], args.size[0]), dtype=np.float32)
        y_batch = np.ndarray((batchSize, args.channels[0], args.size[
                             1], args.size[0]), dtype=np.float32)
        for i in range(0, len(imagelist)):
            print_and_log('frameNo:' + str(i))
            x_batch[0] = read_image(imagelist[i])
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            loss.unchain_backward()
            loss = 0
            if args.gpu >= 0:
                model.to_cpu()
            write_image(x_batch[0].copy(), 'images/test' + str(i) + 'x.jpg')
            write_image(model.y.data[0].copy(),
                        'images/test' + str(i) + 'y.jpg')
            if args.gpu >= 0:
                model.to_gpu()

        if args.gpu >= 0:
            model.to_cpu()
        x_batch[0] = model.y.data[0].copy()
        if args.gpu >= 0:
            model.to_gpu()
        for i in range(len(imagelist), len(imagelist) + args.ext):
            print_and_log('extended frameNo:' + str(i))
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            loss.unchain_backward()
            loss = 0
            if args.gpu >= 0:
                model.to_cpu()
            write_image(model.y.data[0].copy(),
                        'images/test' + str(i) + 'y.jpg')
            x_batch[0] = model.y.data[0].copy()
            if args.gpu >= 0:
                model.to_gpu()


# 訓練のとき
else:
    count = 0
    seq = 0
    loss_sequencelist = 0
    while count < args.period:
        imagelist = load_list(sequencelist[seq], args.root)
        prednet.reset_state()
        loss = 0
        loss_imagelist = 0

        batchSize = 1
        x_batch = np.ndarray((batchSize, args.channels[0], args.size[
                             1], args.size[0]), dtype=np.float32)
        y_batch = np.ndarray((batchSize, args.channels[0], args.size[
                             1], args.size[0]), dtype=np.float32)
        x_batch[0] = read_image(imagelist[0])
        for i in range(1, len(imagelist)):
            y_batch[0] = read_image(imagelist[i])
            loss += model(chainer.Variable(xp.asarray(x_batch)),
                          chainer.Variable(xp.asarray(y_batch)))
            loss_imagelist += loss.data

            print_and_log('frameNo:' + str(i))
            if (i + 1) % args.bprop == 0:
                model.zerograds()
                loss.backward()
                loss.unchain_backward()
                loss = 0
                optimizer.update()
                if args.gpu >= 0:
                    model.to_cpu()
                write_image(x_batch[0].copy(), 'images/' +
                            str(count) + '_' + str(seq) + '_' + str(i) + 'x.jpg')
                write_image(model.y.data[0].copy(
                ), 'images/' + str(count) + '_' + str(seq) + '_' + str(i) + 'y.jpg')
                write_image(y_batch[0].copy(), 'images/' +
                            str(count) + '_' + str(seq) + '_' + str(i) + 'z.jpg')
                if args.gpu >= 0:
                    model.to_gpu()
                print_and_log('loss:' + str(float(model.loss.data)))

            if (count % args.save) == 0:
                print_and_log('save the model')
                serializers.save_npz('models/' + str(count) + '.model', model)
                print_and_log('save the optimizer')
                serializers.save_npz(
                    'models/' + str(count) + '.state', optimizer)

            x_batch[0] = y_batch[0]
            count += 1

        print_and_log('loss-imagelist' + str(seq) + ' : ' + str(loss_imagelist) +
                      ' , lenghth : ' + str(len(imagelist)))
        seq = (seq + 1) % len(sequencelist)
        loss_sequencelist += loss_imagelist
        if seq == len(sequencelist) - 1:
            print_and_log('loss-sequencelist : ' + str(loss_sequencelist))
            loss_sequencelist = 0
