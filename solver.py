""" Solver for Stain Normalization Models

The solver uses stochastic gradient descent with varying learning rates and penalties for each
layer. Configure hyperparameters for all layers in ``hyperparams.json``.

This script is called by

>>> python solver.py [model_id ...]

Currently, four baseline models for stain normalization are implemented, details are given
in ``models.py``. To optimize model 2 and 4, just call

>>> python solver.py 2 4
"""

__author__ = "sschneider"
__email__ = "steffen.schneider@rwth-aachen.de"

import theano
import theano.tensor as T
import json
import logging
import pickle
import time
import os

import numpy as np
from skimage import io

import lasagne as nn

from stainnorm import tools
from stainnorm.tools import get_updates, get_dataset
from stainnorm.models import *
from stainnorm.pca import PCAIterator

DATA_PSIZE = 192

def netdump(layers):
  log = logging.getLogger(__name__)
  log.info("Network Dump")
  for key in layers.keys():
    log.info("{}{}{}".format(key, " " * (20 - len(key)), nn.layers.get_output_shape(layers[key])))
    for param in layers[key].params:
      p = param.get_value()
      fmt = "{}{}{}{:.2f} {:.2f} {:.2f} {:.2f}"
      log.info(fmt.format(" " * 30, str(param), " " * (10 - len(str(param))),
                               p.min(), p.mean(), p.max(), (p ** 2).sum()))

def compile_validation(constructor, psize):
  """ compile network using layer-specific learning rates and penalties
  """
  input_var = T.tensor4("input")

  network, layers = constructor(input_var, input_size=(1,3,psize,psize))

  #test_prediction = nn.layers.get_output(network, deterministic=True, batch_norm_use_averages=False, batch_norm_update_averages=False)
  test_prediction = nn.layers.get_output(network, deterministic=True)
  val_fn = theano.function([input_var], [test_prediction], allow_input_downcast=True)

  return val_fn, network, layers

def compile(constructor, learning_rates, penalties, optimizer=nn.updates.nesterov_momentum):
  """ compile network using layer-specific learning rates and penalties
  """
  input_var = T.tensor4("input")
  target_var = T.tensor4("targets")

  network, layers = constructor(input_var)

  outp = nn.layers.get_output(network)
  netdump(layers)
  if nn.layers.get_output_shape(network)[-1] == tools.INP_PSIZE:
    prediction = tools.crop(outp)
  else:
    prediction = outp

  loss = nn.objectives.squared_error(prediction, target_var)
  loss = loss.flatten().mean()

  reg_layers = {layers[k]: penalties[k] for k in penalties.keys() if k in layers.keys()}
  reg = nn.regularization.regularize_layer_params_weighted(reg_layers, nn.regularization.l2)
  error = loss + reg

  updates, params = get_updates(error, layers, optimizer, learning_rates,
                  lfilter=lambda k: not ("features" in k))

  test_prediction = nn.layers.get_output(network, deterministic=True)
  test_loss = nn.objectives.squared_error(test_prediction, target_var)
  test_loss = test_loss.flatten().mean()

  train_fn = theano.function([input_var, target_var], [error, loss, reg, prediction], updates=updates,
                 allow_input_downcast=True)
  val_fn = theano.function([input_var, target_var], [test_loss, test_prediction], allow_input_downcast=True)

  # dummy functions for testing
  # train_fn = lambda x,y : [0, 0, 0, tools.crop(x)]
  # val_fn = lambda x,y : [0, tools.crop(x)]

  return train_fn, val_fn, network, layers


def run_solver(train_fn, network, layers, experiment_id, timestamp=None,
         save_batch_freq=100, save_time_freq=240, dataset="H.E.T.",
         dir_prefix="run"):
  """ Run a solver given the specified training function

  Parameters
  ----------
  train_fn    : function ``[input_var, target_var] --> [error, loss, reg, prediction]``
  network     : lasagne.layers.Layer representing the output layer
  layers      : dict-like, network layers
  experiment_id : ID for saving of model + losses
  timestamp   : Experiment start time, used for savefolder
  save_batch_freq : int, how often (in terms of batches) a network dump should happen?
  save_time_freq  : int, how often (in terms of seconds) a network dump should happen?

  """
  log = logging.getLogger(__name__)
  if timestamp is None: timestamp = time.time()

  X, Xv = get_dataset(train_key=dataset, val_key=["H.E.T-", "H.E-T.", "H+E+T."])

  regularize_name = lambda s : s.replace("+","p").replace("-","m").replace(".","o")
  savedir = tools.require_dir("{}_{}_{}".format(str(dir_prefix), regularize_name(str(dataset)), str(timestamp)))

  log.info("Loaded dataset with %d training and %d validation images", len(X), len(Xv))

  datagen = PCAIterator(np.concatenate((X[200:], Xv[200:]),axis=0))

  tools.OUT_PSIZE = nn.layers.get_output_shape(network)[2]
  log.info("Setting output size to %d", tools.OUT_PSIZE)

  losses = []
  val_losses = []
  regs = []
  try:
    snap = time.time()
    val_snap = snap
    for epoch in range(20):
      batch = 0
      val_batch = 0
      for inputs, targets in datagen.iterate(X, batch_size=64, shuffle=True):

        ## TRAINING ##
        error, loss, reg, pred = train_fn(inputs, tools.crop(targets))
        pmin, pmean, pmax = pred.min(), pred.mean(), pred.max()
        batch += 1

        ## TRAINING OUTPUT ##
        if batch % save_batch_freq == 0 or time.time() - snap > save_time_freq:
          outp_image = [np.clip(tools.panelize(pp), 0, 255).astype("uint8") for pp in \
                  [tools.crop(inputs), pred, tools.crop(targets)]]
          outp_image = np.concatenate(outp_image, axis=0)
          log.debug("Saving output image: %s", str(outp_image.shape))
          io.imsave(os.path.join(savedir, "model{}_ep{}_b{}.png".format(experiment_id, epoch, batch)),
                outp_image)

          netdump(layers)
          snap = time.time()

        log.info("#{}: loss [{:.3f}, {:.3f}, {:.3f}], outp range [{:.3f}, {:.3f}, {:.3f}]".format(
          batch, float(error), float(loss), float(reg), pmin, pmean, pmax))
        losses.append(loss)
        regs.append(regs)

        ## VALIDATION OUTPUT ##
        if time.time() - val_snap > save_time_freq:
          log.info("Start validation run")
          try:
            for inputs, targets in datagen.iterate(Xv, batch_size=64, shuffle=False, augment=False):
              loss, pred = pred_fn(inputs, tools.crop(targets))
              outp_image = [np.clip(tools.panelize(pp), 0, 255).astype("uint8") for pp in \
                  [tools.crop(inputs), pred, tools.crop(X[0:len(pred)])]]
              outp_image = np.concatenate(outp_image, axis=0)
              log.debug("Loss: %f", loss)
              io.imsave(os.path.join(
                  savedir, "val_model{}_ep{}_b{}.png".format(experiment_id, epoch, val_batch)),
                        outp_image)
              val_batch += 1

              if time.time() - val_snap > save_time_freq //10:
                break
            log.info("Finished validation run")
          except:
            log.exception("Error in validation run. Continue training.")

          val_snap = time.time()

  except KeyboardInterrupt:
    log.info("Done. (User Interrupt)")
  except Exception:
    log.exception("Unhandeled Exception")

  log.info("Saving network")
  weights = nn.layers.get_all_param_values(network)

  data = {"params": weights, "losses": losses, "regularization": regs, "model_id": model_id}
  with open(os.path.join(savedir, str(experiment_id) + ".npz"), "wb") as fp:
    pickle.dump(data, fp)
  log.info("Saved weights and loss history")


def load_config(fname="hyperparams.json"):
  """ Load configuration file with hyperparameters
  """
  with open(fname, "r") as fp:
    config = json.load(fp)
  return config



if __name__ == '__main__':

  import sys
  import argparse
  from stainnorm.tools import logger_setup

  parser = argparse.ArgumentParser(description='Run experiments')
  parser.add_argument("-m", '--model', nargs="+",
        help='Model ID')
  parser.add_argument('-H', '--hematoxylin', type=int, nargs=1, default=1,
        help='control hematoxylin stain (-1/0/1)')
  parser.add_argument('-E', '--eosin', type=int, nargs=1, default=1,
        help='control eosin stain (-1/0/1)')
  parser.add_argument('-T', '--thickness', type=int, nargs=1, default=1,
        help='control thickness (-1/0/1)')
  parser.add_argument('-c', '--comment', type=str, nargs=1, default="run",
        help='comment string, to be added to the save directories')
  args = parser.parse_args()
  mapping = ('-', '.', '+')
  dataset_code = "H{}E{}T{}".format(*[mapping[i[0]] for i in (args.hematoxylin, args.eosin, args.thickness)])

  print("Starting with dataset code {}".format(dataset_code))

  os.makedirs('log', exist_ok=True)
  logger_setup(filename="log/stainnorm.log")
  log = logging.getLogger(__name__)
  log.info("__START__")


  # list of possible models. Model 13, fan_reworked is the model presented in our paper
  # feel free to explore other combinations as well
  modellist =      [build_baseline1_small,                # 1
                    build_baseline2_feats,                # 2
                    build_baseline3_vgg,                  # 3
                    build_baseline4_fan,                  # 4
                    build_baseline5_fan,                  # 5
                    build_baseline6_fan_fan,              # 6
                    build_resnet7_fan,                    # 7
                    build_baseline8_fan_bilinear,         # 8
                    build_baseline9_fan_fan_bilinear,     # 9
                    build_finetuned1_fan,                 #10
                    build_finetuned2_fan,                 #11
                    build_big_fan,                        #12
                    build_fan_reworked]                   #13

  joblist = args.model

  tic = time.time()
  for model_id in joblist:
    model_id = int(model_id) - 1
    if model_id < 0 or model_id >= len(modellist):
      log.warning("Invalid model id. Skipping: %d", model_id)
      continue

    get_model = modellist[int(model_id)]
    log.info("Start processing job id %d for model %s", model_id, str(get_model))
    hyperparams = load_config()
    log.info("Building graph.")
    train_fn, pred_fn, network, layers = compile(get_model, hyperparams["learning_rates"], hyperparams["penalties"])

    log.info("Start with training")
    run_solver(train_fn, network, layers, model_id, timestamp=tic, dataset=dataset_code, dir_prefix=args.comment[0])

  log.info("__DONE__")
