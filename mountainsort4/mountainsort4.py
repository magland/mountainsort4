from typing import Union, cast
from .ms4alg import MountainSort4
import os
import shutil
import tempfile
import numpy as np
import math
import multiprocessing
import spikeextractors as se


def mountainsort4(*, recording: se.RecordingExtractor, detect_sign: int, clip_size: int=50, adjacency_radius: float=-1, detect_threshold: float=3, detect_interval: int=10,
                  num_workers: Union[None, int]=None, verbose: bool=True, use_recording_directly: bool=False, tempdir: Union[str, None]=None) -> se.SortingExtractor:
    if num_workers is None:
        num_workers = math.floor((multiprocessing.cpu_count()+1)/2)

    if verbose:
        print('Using {} workers.'.format(num_workers))

    MS4 = MountainSort4()
    MS4.setRecording(recording)
    geom = _get_geom_from_recording(recording)
    MS4.setGeom(geom)
    MS4.setSortingOpts(
        clip_size=clip_size,
        adjacency_radius=adjacency_radius,
        detect_sign=detect_sign,
        detect_interval=detect_interval,
        detect_threshold=detect_threshold,
        verbose=verbose
    )

    if tempdir is None:
        tempdir = tempfile.mkdtemp(dir=os.environ.get('TEMPDIR', '/tmp'))
    else:
        tempdir = tempfile.mkdtemp(dir=tempdir)
        
    MS4.setNumWorkers(num_workers)
    if verbose:
        print('Using tempdir: '+tempdir)
    MS4.setTemporaryDirectory(tempdir)
    MS4.setUseRecordingDirectly(use_recording_directly)
    try:
        MS4.sort()
    except:
        if verbose:
            print('Cleaning tempdir:: '+tempdir)
        shutil.rmtree(tempdir)
        raise
    if verbose:
        print('Cleaning tempdir::::: '+tempdir)
    shutil.rmtree(tempdir)
    times, labels, channels = MS4.eventTimesLabelsChannels()
    output = se.NumpySortingExtractor()
    output.set_times_labels(times=times, labels=labels)
    return output


def _get_geom_from_recording(recording: se.RecordingExtractor):
    channel_ids = cast(np.ndarray, recording.get_channel_ids())
    M = len(channel_ids)
    location0 = recording.get_channel_property(channel_ids[0], 'location')
    nd = len(location0)
    geom = np.zeros((M, nd))
    for i in range(M):
        location_i = recording.get_channel_property(channel_ids[i], 'location')
        geom[i, :] = location_i
    return geom
