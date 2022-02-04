import logging
import time
import numpy as np

import grpc

from .federated_learning_pb2 import Header, WeightRequest
from .federated_learning_pb2 import DataBuffer
from .federated_learning_pb2 import JobRequest
from .federated_learning_pb2 import LearningResults
from .federated_learning_pb2 import TensorRequest
from .federated_learning_pb2 import TensorRecord
from .federated_learning_pb2 import WeightRequest
from .federated_learning_pb2_grpc import FederatedLearningStub
from . import utils

class FLClient():
    def __init__(self, client_id, server_uri, max_message_size=2*1024*1024):
        self.client_id = client_id
        self.max_message_size = max_message_size
        self.channel = grpc.insecure_channel(
            server_uri,
            options=[
                ('grpc.max_send_message_length', max_message_size),
                ('grpc.max_receive_message_length', max_message_size)
            ]
        )
        grpc.channel_ready_future(self.channel).result(timeout=60)
        self.stub = FederatedLearningStub(self.channel)
        self.header = Header(server_id=1, client_id=self.client_id)
        self.logger = logging.getLogger(__name__)
        self.time_get_job = 0.0
        self.time_get_tensor = 0.0
        self.time_send_results = 0.0

    def get_job(self, job_done):
        request = JobRequest(header=self.header, job_done=job_done)
        start = time.time()
        response = self.stub.GetJob(request)
        end = time.time()
        self.time_get_job += end - start
        self.logger.info(f"[Client ID: {self.client_id: 03}] Received JobReponse with (server,round,job)=(%d,%d,%d)",
                         response.header.server_id, response.round_number, response.job_todo)
        return response.round_number, response.job_todo

    def get_tensor_record(self, name, round_number):
        request = TensorRequest(header=self.header, name=name, round_number=round_number)
        start = time.time()
        response = self.stub.GetTensorRecord(request)
        end = time.time()
        self.time_get_tensor += end - start
        shape = tuple(response.data_shape)
        flat = np.frombuffer(response.data_bytes, dtype=np.float32)
        nparray = np.reshape(flat, newshape=shape, order='C')
        return nparray

    def get_weight(self, training_size):
        request = WeightRequest(header=self.header, size=training_size)
        response = self.stub.GetWeight(request)
        return response.weight

    def send_learning_results(self, penalty, primal, dual, round_number):
        primal_tensors = [utils.construct_tensor_record(k, np.array(v)) for k,v in primal.items()]
        dual_tensors = [utils.construct_tensor_record(k, np.array(v)) for k,v in dual.items()]
        proto = LearningResults(header=self.header, round_number=round_number, penalty=penalty[self.client_id], primal=primal_tensors, dual=dual_tensors)

        databuffer = []
        databuffer += utils.proto_to_databuffer(proto, max_message_size=self.max_message_size)
        start = time.time()
        self.stub.SendLearningResults(iter(databuffer))
        end = time.time()
        self.time_send_results += end - start

    def get_comm_time(self):
        return self.time_get_job+self.time_get_tensor+self.time_send_results

