"""Client-end for the ASR demo."""
import keyboard
import struct
import socket
import sys
import argparse
import pyaudio
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--host_ip",
    default="localhost",
    type=str,
    help="Server IP address. (default: %(default)s)")
parser.add_argument(
    "--host_port",
    default=8086,
    type=int,
    help="Server Port. (default: %(default)s)")
args = parser.parse_args()

is_recording = False
enable_trigger_record = True


def on_press_release(x):
    """Keyboard callback function."""
    global is_recording, enable_trigger_record, data_list, path
    press = keyboard.KeyboardEvent('down', 28, 'space')
    release = keyboard.KeyboardEvent('up', 28, 'space')
    if x.event_type == 'down' and x.name == press.name:
        if (not is_recording) and enable_trigger_record:
            sys.stdout.write("Start Recording ... ")
            sys.stdout.flush()
            is_recording = True
    if x.event_type == 'up' and x.name == release.name:
        if is_recording == True:
            is_recording = False
            print("\nfinish recording")
            write_wave()

import wave
import threading

data_list = []

class PrintThread(threading.Thread):
    def __init__(self, socket):
        threading.Thread.__init__(self)
        self.socket = socket

    def run(self):
        print("Start Print thread")
        received = self.socket.recv(1024)
        time.sleep(0.01)
        count = 3
        while len(received) != 0 or count > 0:
            if len(received) != 0:
                print("*" + received.decode("utf-8") + "*")
            else:
                count -= 1
                time.sleep(0.5)
            received = self.socket.recv(1024)
        print("End Print thread")
        self.socket.close()


class Callback(object):
    def __init__(self):
        self.sock = None

    def callback(self,in_data, frame_count, time_info, status):
        """Audio recorder's stream callback function."""
        global data_list, is_recording, enable_trigger_record
        if is_recording:
            if enable_trigger_record:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((args.host_ip, args.host_port))
                self.pt = PrintThread(self.sock)
                self.pt.start()
            enable_trigger_record = False
            if len(in_data) > 0:
                start = time.time()
                self.send_data(in_data, frame_count)
                end = time.time()
        else:
            if not enable_trigger_record:
                self.send_data(in_data, frame_count)
                time.sleep(0.01)
                self.send_data(b'', frame_count)
                self.pt.join()
                enable_trigger_record = True
        return (in_data, pyaudio.paContinue)

    def send_data(self, in_data, frame_count):
        if len(in_data) == 0:
            print("send zero data")
        data_list.append(in_data)
        all = struct.pack('>i', len(in_data))+in_data
        print("send data length", len(all))
        self.sock.sendall(all)


def write_wave(CHANNELS=1,RATE=16000 ):
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data_list))
    wf.close()

def main():
    # prepare audio recorder
    callback = Callback().callback

    # duration of signal frame, seconds
    FRAME_LEN = 0.2  # 0.2
    # number of audio channels (expect mono signal)
    CHANNELS = 1
    # sample rate, Hz
    RATE = 16000

    CHUNK_SIZE = int(FRAME_LEN * RATE)

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        stream_callback=callback,
        frames_per_buffer=CHUNK_SIZE)
    stream.start_stream()

    # prepare keyboard listener
    while (1):
        keyboard.hook(on_press_release)
        if keyboard.record('esc'):
            break

    # close up
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()
