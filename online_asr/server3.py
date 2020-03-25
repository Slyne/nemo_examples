"""Server-end for the ASR demo."""
import os
import time
import random
import argparse
import functools
from time import gmtime, strftime
import socketserver
import struct
import wave
import numpy as np
import distutils.util
import threading
from frame import FrameASR
from multiprocessing import Queue, Process
from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
from model import model_definition, Model

model = Model(model_definition)
lock = threading.Lock()
input_queue = Queue()
output_queue = Queue()

def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.
    Usage:
    .. code-block:: python
        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('host_port',        int,    8086,    "Server's IP port.")
add_arg('host_ip',          str,
        '10.19.203.82',
        "Server's IP address.")
add_arg('frame_len',        float,    0.2,    "frame length in senconds")
add_arg('frame_overlap',    float,    2,      "frame overlapping in seconds")
add_arg('offset',           float,    0,      "offset in seconds")

add_arg('decoder_type',          str,   
        'ctc_beam_search',
        "greedy_search or ctc_beam_search")
args = parser.parse_args()


class AsrTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """The ASR TCP Server."""

    def __init__(self,
                 server_address,
                 RequestHandlerClass,
                 bind_and_activate=True,
                 ):
        socketserver.TCPServer.__init__(
            self, server_address, RequestHandlerClass, bind_and_activate=True)


class AsrRequestHandler(socketserver.BaseRequestHandler):
    """The ASR request handler.""" 

    def whether_stop_receive(self, data, chunk_size):
        
        if len(data) % (chunk_size) != 0:
            last_chunk_index = -(len(data)%(chunk_size))
        else:
            last_chunk_index = -chunk_size

        last_chunk = data[last_chunk_index:]
        last_chunk_size = struct.unpack('>i', last_chunk[:4])[0]
        if last_chunk_size == 0:
            return False
        else:
            return True

    def handle(self):

        #self.request.settimeout(1)
        asr = FrameASR(model_definition, decoder_type=args.decoder_type,
               frame_len=args.frame_len, frame_overlap=args.frame_overlap, 
               offset=args.offset)
    
        cur_thread = threading.current_thread().name
        if cur_thread not in thread_resultQueue:
            thread_resultQueue[cur_thread] = queue.Queue()

        data = b''
        receive_data_mark = True
        close_count = 5

        model_target_len = asr.n_frame_len * 2 
        chunk_size = model_target_len + 4

        while True:
            if receive_data_mark:
                chunk = self.request.recv(1024) 
                data += chunk      
                
                receive_data_mark = self.whether_stop_receive(data, chunk_size)

                while len(data) < chunk_size and receive_data_mark:
                    data += self.request.recv(1024) 

                receive_data_mark = self.whether_stop_receive(data, chunk_size)

            in_data = data[4:model_target_len+4]
            data = data[model_target_len + 4:]
            
            if len(in_data) == 0:
                print("IN DATA IS ZERO, begin to close, ", close_count)
                if close_count == 0:
                    break
                else:
                    time.sleep(0.1)
                    close_count -= 1
            else:
                signal = np.frombuffer(in_data, dtype=np.int16)
                pad_signal = asr.process_signal(signal)              
                input_queue.put((cur_thread, pad_signal))

            if not thread_resultQueue[cur_thread].empty():
                logits = thread_resultQueue[cur_thread].get()
                start_time = time.time()
                transcript = asr.post_process(logits)
                finish_time = time.time()
                print("Response Time: %f, *Transcript: %s *" %
                    (finish_time - start_time, transcript))
                if transcript == "":
                    transcript = "<blank>"
                self.request.sendall(transcript.encode('utf-8'))

        del thread_resultQueue[cur_thread]


def start_server():
    """Start the ASR server"""

    # start the server
    server = AsrTCPServer(
        server_address=(args.host_ip, args.host_port),
        RequestHandlerClass=AsrRequestHandler)
    print("ASR Server Started.")
    server.serve_forever()


def run(input_queue, output_queue, batch_size=8):
    print("Start model process")
    while True:
        batch_signal = []
        batch_thread = []         
        while not input_queue.empty() and len(batch_signal) < batch_size:
            thread, signal = input_queue.get()
            batch_signal.append(signal)
            batch_thread.append(thread)
        
        if len(batch_signal) != 0:
            logits = model.infer_signal(np.asarray(batch_signal)).cpu().numpy()     
            for thread, logit in zip(batch_thread, logits):
                output_queue.put((thread,logit))
    print("Close model process")


class ModelResultDispatch(threading.Thread):
    def __init__(self, output_queue):
        threading.Thread.__init__(self)
        self.output_queue = output_queue

    def run(self):
        print("start dispatch thread")
        while True:
            while not self.output_queue.empty():
                thread, logit = self.output_queue.get()
                thread_resultQueue[thread].put(logit)
        print("close dispatch thread")

import queue
thread_resultQueue = {}


def print_arguments(args):
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(vars(args).items()):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")

def main():
    print_arguments(args)
    model_process = Process(target=run, args=(input_queue, output_queue))
    model_process.start()
    model_dispatch_thread = ModelResultDispatch(output_queue)
    model_dispatch_thread.start()
    start_server()
    model_process.join()
    model_dispatch_thread.join()


if __name__ == "__main__":
    main()
