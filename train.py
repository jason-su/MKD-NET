#coding:utf-8
#!/usr/bin/env python
import os
import json
import torch
import numpy as np
import queue
import pprint
import random 
import argparse
import importlib
import threading
import traceback

from tqdm import tqdm
from torch.multiprocessing import Process, Queue, Pool

from core.dbs import datasets
from core.utils import stdout_to_tqdm
from core.config import SystemConfig
from core.sample import data_sampling_func
from core.nnet.py_factory import NetworkFactory
from tensorboardX import SummaryWriter

writer = SummaryWriter('logs/merge-class-voc-1123/')

torch.backends.cudnn.enabled   = True
torch.backends.cudnn.benchmark = True

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument("--cfg_file", help="config file", default="CornerNet_Squeeze", type=str)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--initialize", action="store_true")
    

    args = parser.parse_args()
    return args

#call sample_data (data_sampling_func) method to read data
def prefetch_data(system_config, db, queue, sample_data, data_aug):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(system_config, db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e

def _pin_memory(ts):
    if type(ts) is list:
        return [t.pin_memory() for t in ts]
    return ts.pin_memory()

#pinned_data_queue only contain xs and ys
def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [_pin_memory(x) for x in data["xs"]]
        data["ys"] = [_pin_memory(y) for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return

def init_parallel_jobs(system_config, dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(system_config, db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks

def terminate_tasks(tasks):
    for task in tasks:
        task.terminate()

def train(training_dbs, validation_db, system_config, model, args):
    # reading arguments from command
    start_iter  = args.start_iter
    initialize  = args.initialize
    gpu         = args.gpu

    # reading arguments from json file
    learning_rate    = system_config.learning_rate
    max_iteration    = system_config.max_iter
    pretrained_model = system_config.pretrain
    stepsize         = system_config.stepsize
    snapshot         = system_config.snapshot
    val_iter         = system_config.val_iter
    display          = system_config.display
    decay_rate       = system_config.decay_rate

    print("building model...")
    nnet = NetworkFactory(system_config, model, gpu=gpu)
    if initialize:
        nnet.save_params(0)
        exit(0)

    # queues storing data for training
    training_queue   = Queue(system_config.prefetch_size)
    validation_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue   = queue.Queue(system_config.prefetch_size)
    pinned_validation_queue = queue.Queue(5)

    # allocating resources for parallel reading
    # parallel read train data to queue
    # 每个worker对应一份training_db，生成workder个并行读数据的进程
    training_tasks = init_parallel_jobs(system_config, training_dbs, training_queue, data_sampling_func, True)
    if val_iter:
        validation_tasks = init_parallel_jobs(system_config, [validation_db], validation_queue, data_sampling_func, False)

    #设置进程信号量，线程负责把数据从training_queue读到pinned_training_queue中
    training_pin_semaphore   = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()
 
    training_pin_args   = (training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()
 
    validation_pin_args   = (validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)

    if start_iter:
        nnet.load_params(start_iter)
        learning_rate /= (decay_rate ** (start_iter // stepsize))
        learning_rate = max(5e-5,learning_rate)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)

    print("training start, max iteration {}".format(max_iteration))
        
    nnet.cuda()
    nnet.train_mode()
    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=80):
#         for iteration in range(start_iter + 1, max_iteration + 1):
            training = pinned_training_queue.get(block=True)
            training_loss,focal_loss,pull_loss,push_loss,off_loss = nnet.train(training["xs"],training["ys"])
            
#             if display and iteration % display == 0:
#                 print("training loss at iteration {}: {}".format(iteration, training_loss.item()))
#                 
#             print("[log-loss]:{}={}".format(iteration, training_loss.item()))
            
            writer.add_scalar('train_loss', training_loss, global_step=iteration)
            writer.add_scalar('focal_loss', focal_loss, global_step=iteration)
            writer.add_scalar('pull_loss', pull_loss, global_step=iteration)
            writer.add_scalar('push_loss', push_loss, global_step=iteration)
            writer.add_scalar('off_loss', off_loss, global_step=iteration)
            
            del training_loss

            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                nnet.eval_mode()
                validation = pinned_validation_queue.get(block=True)
                validation_loss = nnet.validate(validation["xs"],validation["ys"])
                print("[log-validation-loss]:{}={}".format(iteration, validation_loss.item()))
                writer.add_scalar('validation_loss', validation_loss, global_step=iteration)
                
                nnet.train_mode()

            if iteration % snapshot == 0:
                nnet.save_params(iteration)

            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                learning_rate = max(5e-5,learning_rate)
                nnet.set_lr(learning_rate)
                print("set learning rate {}".format(learning_rate))


    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()

    # terminating data fetching processes
    terminate_tasks(training_tasks)
    terminate_tasks(validation_tasks)
    
    writer.close()

def main(args):
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    args.gpu = None
    
    cfg_file = os.path.join("./configs", args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        config = json.load(f)
    
    print("load cfg file: {}".format(cfg_file))

    #update fields in config.py by the json file
    config["system"]["snapshot_name"] = args.cfg_file
    system_config = SystemConfig().update_config(config["system"])

    #init model according the config file name
    model_file  = "core.models.{}".format(args.cfg_file)
    model_file  = importlib.import_module(model_file)
    model       = model_file.model()
    
    #set train and val dataset name
    train_split = system_config.train_split
    val_split   = system_config.val_split

    print("loading all datasets...")
    dataset = system_config.dataset
    workers = args.workers
    print("using {} workers".format(workers))
    training_dbs = [datasets[dataset](config["db"], split=train_split, sys_config=system_config) for _ in range(workers)]
    validation_db = datasets[dataset](config["db"], split=val_split, sys_config=system_config)

    print("system config...")
    pprint.pprint(system_config.full)

    print("db config...")
    pprint.pprint(training_dbs[0].configs)

    print("len of db: {}".format(len(training_dbs[0].db_inds)))

    train(training_dbs, validation_db, system_config, model, args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
