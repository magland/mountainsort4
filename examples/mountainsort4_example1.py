#!/usr/bin/env python3

import mountainsort4 as ms4
import spikeinterface.extractors as se

def main():
    recording, sorting_true = se.toy_example(num_segments=1)
    sorting = ms4.mountainsort4(
        recording=recording,
        detect_sign=-1,
        clip_size=50,
        adjacency_radius=20,
        detect_threshold=3,
        detect_interval=10,
        num_workers=None,
        verbose=True
    )
    print(f'Found {len(sorting.get_unit_ids())} units')


if __name__ == '__main__':
    main()
