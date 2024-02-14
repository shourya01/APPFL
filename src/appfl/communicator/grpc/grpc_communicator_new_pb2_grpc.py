# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import grpc_communicator_new_pb2 as grpc__communicator__new__pb2


class NewGRPCCommunicatorStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetConfiguration = channel.unary_unary(
                '/NewGRPCCommunicator/GetConfiguration',
                request_serializer=grpc__communicator__new__pb2.ConfigurationRequest.SerializeToString,
                response_deserializer=grpc__communicator__new__pb2.ConfigurationResponse.FromString,
                )
        self.GetGlobalModel = channel.unary_stream(
                '/NewGRPCCommunicator/GetGlobalModel',
                request_serializer=grpc__communicator__new__pb2.GlobalModelRequest.SerializeToString,
                response_deserializer=grpc__communicator__new__pb2.DataBufferNew.FromString,
                )
        self.SendLocalModel = channel.stream_stream(
                '/NewGRPCCommunicator/SendLocalModel',
                request_serializer=grpc__communicator__new__pb2.DataBufferNew.SerializeToString,
                response_deserializer=grpc__communicator__new__pb2.DataBufferNew.FromString,
                )
        self.CustomAction = channel.unary_unary(
                '/NewGRPCCommunicator/CustomAction',
                request_serializer=grpc__communicator__new__pb2.CustomActionRequest.SerializeToString,
                response_deserializer=grpc__communicator__new__pb2.CustomActionResponse.FromString,
                )


class NewGRPCCommunicatorServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetConfiguration(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetGlobalModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SendLocalModel(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CustomAction(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_NewGRPCCommunicatorServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetConfiguration': grpc.unary_unary_rpc_method_handler(
                    servicer.GetConfiguration,
                    request_deserializer=grpc__communicator__new__pb2.ConfigurationRequest.FromString,
                    response_serializer=grpc__communicator__new__pb2.ConfigurationResponse.SerializeToString,
            ),
            'GetGlobalModel': grpc.unary_stream_rpc_method_handler(
                    servicer.GetGlobalModel,
                    request_deserializer=grpc__communicator__new__pb2.GlobalModelRequest.FromString,
                    response_serializer=grpc__communicator__new__pb2.DataBufferNew.SerializeToString,
            ),
            'SendLocalModel': grpc.stream_stream_rpc_method_handler(
                    servicer.SendLocalModel,
                    request_deserializer=grpc__communicator__new__pb2.DataBufferNew.FromString,
                    response_serializer=grpc__communicator__new__pb2.DataBufferNew.SerializeToString,
            ),
            'CustomAction': grpc.unary_unary_rpc_method_handler(
                    servicer.CustomAction,
                    request_deserializer=grpc__communicator__new__pb2.CustomActionRequest.FromString,
                    response_serializer=grpc__communicator__new__pb2.CustomActionResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'NewGRPCCommunicator', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class NewGRPCCommunicator(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetConfiguration(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NewGRPCCommunicator/GetConfiguration',
            grpc__communicator__new__pb2.ConfigurationRequest.SerializeToString,
            grpc__communicator__new__pb2.ConfigurationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetGlobalModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/NewGRPCCommunicator/GetGlobalModel',
            grpc__communicator__new__pb2.GlobalModelRequest.SerializeToString,
            grpc__communicator__new__pb2.DataBufferNew.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SendLocalModel(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/NewGRPCCommunicator/SendLocalModel',
            grpc__communicator__new__pb2.DataBufferNew.SerializeToString,
            grpc__communicator__new__pb2.DataBufferNew.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CustomAction(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/NewGRPCCommunicator/CustomAction',
            grpc__communicator__new__pb2.CustomActionRequest.SerializeToString,
            grpc__communicator__new__pb2.CustomActionResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
