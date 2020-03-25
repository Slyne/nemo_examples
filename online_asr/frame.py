# class for streaming frame-based ASR
# 1) use reset() method to reset FrameASR's state
# 2) call transcribe(frame) to do ASR on
#    contiguous signal's frames
import numpy as np
from online_decoder import OnlineDecoder

class FrameASR:
    
    def __init__(self, model_definition, decoder_type,
                 frame_len=2, frame_overlap=2.5, 
                 offset=10):
        '''
        Args:
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          offset: number of symbols to drop for smooth streaming
        '''
        self.vocab = list(model_definition['labels'])
        self.vocab.append('_')
        
        self.sr = model_definition['sample_rate']
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * self.sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * self.sr)
        timestep_duration = model_definition['AudioToMelSpectrogramPreprocessor']['window_stride']
        for block in model_definition['JasperEncoder']['jasper']:
            timestep_duration *= block['stride'][0] ** block['repeat']
        
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration)  # -2
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len,
                               dtype=np.float32)
        self.offset = offset
        self.decoder_type = decoder_type
        self._decoder = OnlineDecoder(len(self.vocab)-1, self.vocab, decoder_type)
        self.reset()

        self.n_timesteps_frame = int(frame_len/timestep_duration)
        print("frame steps:", self.n_timesteps_frame)

    
    def post_process(self, logits, merge=True, if_last=False):
        # print(logits.shape)
        #decoded = self._decoder.decode(
        #    logits[self.n_timesteps_overlap:-self.n_timesteps_overlap],
        #    merge=merge
        #)
        # two timesteps look ahead
        decoded = self._decoder.decode(
            logits[self.n_timesteps_overlap*2-2*self.n_timesteps_frame : -2*self.n_timesteps_frame],
            merge=merge
        )
        return decoded[:len(decoded)-self.offset]
    
    def process_signal(self, frame=None):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            # pad from the end
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        
        return self.buffer
    
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        self._decoder.reset(self.decoder_type)