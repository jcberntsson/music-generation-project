import pretty_midi
import numpy as np
import os

def piano_roll_to_pretty_midi(piano_roll, fs=100, program=0):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    return pm

def piano_roll_to_midi(mat, fn, fs, volume=75):
    """
    Wrapper function for piano_roll_to_pretty_midi
    """
    mat = mat.copy()
    mat[mat < 0.5] = 0
    mat[mat > 0] = 1
    pm = piano_roll_to_pretty_midi(mat * volume, fs=fs, program=0)
    pm.write(fn)

def generate_combine_piano_roll(midi_data, fs):
    """
    Generates a piano roll from the provided midi_data with sampling frequency fs.
    Note that if instruments does not match eachother in dimension, padding will be used.
    """
    mat = None
    for inst in midi_data.instruments:
        inst.remove_invalid_notes()
        if inst.is_drum == False:# and inst.program == 0:
            inst_mat = inst.get_piano_roll(fs=fs)
            if mat is None:
                mat = inst_mat
            elif inst_mat.shape == mat.shape:
                mat += inst_mat
            elif inst_mat.shape[1] < mat.shape[1]:
                inst_mat = np.pad(inst_mat, ((0,0),(0,mat.shape[1]-inst_mat.shape[1])), mode='constant')
                mat += inst_mat
            else:
                mat = np.pad(mat, ((0,0),(0,inst_mat.shape[1] - mat.shape[1])), mode='constant')
                mat += inst_mat
    
    if mat is None:
        return None
    
    mat[mat > 0] = 1
    return mat.T

def split_sample(sample, length, n_samples=None):
    """
    Split the provided piano roll into several defined by length.
    """
    (time_steps, pitch_level) = sample.shape
    if n_samples == None:
        n_samples = int(time_steps / length)
    samples = np.zeros((n_samples, length, pitch_level))
    max_start = time_steps - length
    for i in range(0, n_samples):
        start = int(i * max_start / n_samples)
        end = start + length
        samples[i] = sample[start:end, :]
    return samples
