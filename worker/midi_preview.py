"""Render a bounded preview from actual MIDI timing, without external soundfonts."""
import math
from pathlib import Path
import mido
import numpy as np
import soundfile as sf

def render_preview(source, target):
    midi = mido.MidiFile(source)
    time = 0.0
    pending = {}
    notes = []
    for message in midi:
        time += message.time
        if time > 90:
            break
        if message.type == 'note_on' and message.velocity:
            pending[(message.channel,message.note)] = (time,message.velocity)
        elif message.type in {'note_off','note_on'}:
            start = pending.pop((message.channel,message.note),None)
            if start:
                notes.append((message.note,start[0],time,start[1]))
    for (_channel,pitch),(start,velocity) in pending.items():
        notes.append((pitch,start,min(90,time+.2),velocity))
    notes=notes[:3000]
    if not notes:
        return []
    rate=22050
    audio=np.zeros(int((max(note[2] for note in notes)+.25)*rate),dtype=np.float32)
    for pitch,start,end,velocity in notes:
        length=max(1,int((end-start)*rate)); times=np.arange(length)/rate; frequency=440*2**((pitch-69)/12)
        tone=(np.sin(2*math.pi*frequency*times)+.2*np.sin(4*math.pi*frequency*times))*np.exp(-times*1.8)
        envelope=np.minimum(1,np.minimum(times/.008,(length/rate-times)/.06))
        offset=int(start*rate); audio[offset:offset+length]+=tone*envelope*(velocity/127)*.16
    peak=np.max(np.abs(audio))
    if peak>.95:
        audio*=.95/peak
    sf.write(target,audio,rate,subtype='PCM_16')
    return [{'midi_note':pitch,'note_name':mido.get_note_name(pitch) if hasattr(mido,'get_note_name') else ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][pitch%12]+str(pitch//12-1),'onset_sec':round(start,3),'duration_sec':round(end-start,3),'velocity':velocity} for pitch,start,end,velocity in notes]
