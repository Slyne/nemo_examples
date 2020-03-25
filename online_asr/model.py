import nemo
import nemo.collections.asr as nemo_asr
import numpy as np
import time
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import AudioSignal,LengthsType
import torch

MODEL_YAML = '/home/slyned/projects/nemo/examples/asr/configs/quartznet15x5.yaml'
CHECKPOINT_ENCODER = '../quartznet15x5/JasperEncoder-STEP-247400.pt'
CHECKPOINT_DECODER = '../quartznet15x5/JasperDecoderForCTC-STEP-247400.pt'

from ruamel.yaml import YAML
yaml = YAML(typ="safe")
with open(MODEL_YAML) as f:
    model_definition = yaml.load(f)

class Model(object):
    def __init__(self, model_definition):
        self.model_definition = model_definition
        # some changes for streaming scenario
        self.model_definition['AudioToMelSpectrogramPreprocessor']['dither'] = 0
        self.model_definition['AudioToMelSpectrogramPreprocessor']['pad_to'] = 0
        # spectrogram normalization constants
        normalization = {}
        normalization['fixed_mean'] = [
            -14.95827016, -12.71798736, -11.76067913, -10.83311182,
            -10.6746914,  -10.15163465, -10.05378331, -9.53918999,
            -9.41858904,  -9.23382904,  -9.46470918,  -9.56037,
            -9.57434245,  -9.47498732,  -9.7635205,   -10.08113074,
            -10.05454561, -9.81112681,  -9.68673603,  -9.83652977,
            -9.90046248,  -9.85404766,  -9.92560366,  -9.95440354,
            -10.17162966, -9.90102482,  -9.47471025,  -9.54416855,
            -10.07109475, -9.98249912,  -9.74359465,  -9.55632283,
            -9.23399915,  -9.36487649,  -9.81791084,  -9.56799225,
            -9.70630899,  -9.85148006,  -9.8594418,   -10.01378735,
            -9.98505315,  -9.62016094,  -10.342285,   -10.41070709,
            -10.10687659, -10.14536695, -10.30828702, -10.23542833,
            -10.88546868, -11.31723646, -11.46087382, -11.54877829,
            -11.62400934, -11.92190509, -12.14063815, -11.65130117,
            -11.58308531, -12.22214663, -12.42927197, -12.58039805,
            -13.10098969, -13.14345864, -13.31835645, -14.47345634]

        normalization['fixed_std'] = [
            3.81402054, 4.12647781, 4.05007065, 3.87790987,
            3.74721178, 3.68377423, 3.69344,    3.54001005,
            3.59530412, 3.63752368, 3.62826417, 3.56488469,
            3.53740577, 3.68313898, 3.67138151, 3.55707266,
            3.54919572, 3.55721289, 3.56723346, 3.46029304,
            3.44119672, 3.49030548, 3.39328435, 3.28244406,
            3.28001423, 3.26744937, 3.46692348, 3.35378948,
            2.96330901, 2.97663111, 3.04575148, 2.89717604,
            2.95659301, 2.90181116, 2.7111687,  2.93041291,
            2.86647897, 2.73473181, 2.71495654, 2.75543763,
            2.79174615, 2.96076456, 2.57376336, 2.68789782,
            2.90930817, 2.90412004, 2.76187531, 2.89905006,
            2.65896173, 2.81032176, 2.87769857, 2.84665271,
            2.80863137, 2.80707634, 2.83752184, 3.01914511,
            2.92046439, 2.78461139, 2.90034605, 2.94599508,
            2.99099718, 3.0167554,  3.04649716, 2.94116777]
            
        self.model_definition['AudioToMelSpectrogramPreprocessor']['normalize'] = normalization
        self.neural_factory = nemo.core.NeuralModuleFactory(
            placement=nemo.core.DeviceType.GPU,
            backend=nemo.core.Backend.PyTorch)

        self.data_layer = AudioDataLayer(self.model_definition["sample_rate"])
        self. data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
            **self.model_definition['AudioToMelSpectrogramPreprocessor'])

        self.jasper_encoder = nemo_asr.JasperEncoder(
            feat_in=self.model_definition['AudioToMelSpectrogramPreprocessor']['features'],
            **self.model_definition['JasperEncoder'])

        self.jasper_decoder = nemo_asr.JasperDecoderForCTC(
            feat_in=self.model_definition['JasperEncoder']['jasper'][-1]['filters'],
            num_classes=len(model_definition['labels']))


        self.load_model(CHECKPOINT_ENCODER, CHECKPOINT_DECODER)
        self.create_dag()

    def load_model(self, checkpoint_encoder, checkpoint_decoder):
        # load pre-trained model
        self.jasper_encoder.restore_from(checkpoint_encoder)
        self.jasper_decoder.restore_from(checkpoint_decoder)
    
    def create_dag(self):
        # Define inference DAG
        audio_signal, audio_signal_len = self.data_layer()
        processed_signal, processed_signal_len = self.data_preprocessor(
            input_signal=audio_signal,
            length=audio_signal_len)
        encoded, _ = self.jasper_encoder(audio_signal=processed_signal,
                                            length=processed_signal_len)
        self.log_probs = self.jasper_decoder(encoder_output=encoded)
    
    def infer_signal(self, signal):
        # inference method for audio signal   
        batch_size = len(signal)
        self.data_layer.set_signal(signal, batch_size)
        tensors = self.neural_factory.infer([self.log_probs], verbose=False)
        logits = tensors[0][0]
        return logits  

# simple data layer to pass audio signal
class AudioDataLayer(DataLayerNM):
    @property
    def output_ports(self):
        return {
            "input_signal": NeuralType(('B', 'T'), AudioSignal(freq=self.sample_rate)),
            "length": NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self.output = True
        self.sample_rate = sample_rate
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
            torch.as_tensor(self.signal_shape, dtype=torch.int64)
    
    def set_signal(self, signal, batch_size):
        self.signal = np.reshape(signal.astype(np.float32)/32768., [batch_size, -1])
        self.signal_shape = np.ones(batch_size, dtype=np.int64)*self.signal.shape[1] 
        self.output = True

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self
