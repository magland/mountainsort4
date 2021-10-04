from typing import Union, cast
from .ms4alg import MountainSort4
import os
import shutil
import tempfile
import numpy as np
import math
import multiprocessing
import spikeinterface as si


def mountainsort4(*, recording: si.BaseRecording, detect_sign: int, clip_size: int=50, adjacency_radius: float=-1, detect_threshold: float=3, detect_interval: int=10,
                  num_workers: Union[None, int]=None, verbose: bool=True) -> si.BaseSorting:
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
    tmpdir = tempfile.mkdtemp(dir=os.environ.get('TEMPDIR', '/tmp'))
    MS4.setNumWorkers(num_workers)
    if verbose:
        print('Using tmpdir: '+tmpdir)
    MS4.setTemporaryDirectory(tmpdir)
    try:
        MS4.sort()
    except:
        if verbose:
            print('Cleaning tmpdir:: '+tmpdir)
        shutil.rmtree(tmpdir)
        raise
    if verbose:
        print('Cleaning tmpdir::::: '+tmpdir)
    shutil.rmtree(tmpdir)
    times, labels, channels = MS4.eventTimesLabelsChannels()
    output = si.NumpySorting.from_times_labels(times_list=times, labels_list=labels,
                                               sampling_frequency=recording.get_sampling_frequency())
    return output


def _get_geom_from_recording(recording: si.BaseRecording):
    if 'location' in recording.get_property_keys():
        geom = recording.get_channel_locations()
    else:
        raise AttributeError("mountainsort4 needs locations to be added to the recording object")
    return geom
