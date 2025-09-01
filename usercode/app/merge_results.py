'''
Merge results that came from multiple different files (meaning multiple different executions, BUT FOR THE SAME EXPERIMENT)
The results we merge are mainly summary lists and times, as those are the only ones that change across runs, EVEN FOR THE SAME ARGUMENTS
Files that belong to different queries, experiments or real-time parameters SHOULD NOT be merged.
Doing so is considered UNDEFINED BEHAVIOR 🗣

For jointly evaluating (in a single average) executions that came from the same experiment, but different queries, use the + symbol in eval.py
'''

import os
import sys
sys.path.append(os.path.abspath("../.."))

import argparse
import pickle

#=================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-f", "--files", required=True, action="store", type=str, help="Pickle files to load")
    parser.add_argument("-n", "--name", required=True, action="store", type=str, help="New filename for merge")
    parser.add_argument("--delete-original", action="store_true", default=False, help="Delete partial files during a merge operation")

    args = parser.parse_args()

    #-----------------------------------------------------------------------------------
    merged = None
    partial_files = []

    for file in args.files.split(","):
        partial_files.append(file)
        with open(file, "rb") as f:
            temp = pickle.load(f)
    
        if not isinstance(temp['times'], list):
            temp['times'] = [temp['times']]

        if merged is None:
            merged = temp
        else:
            #Merge times
            merged['times'] += temp['times']

            #Merge summaries
            if temp['summaries'][0]['summary'] is None:
                continue
            elif merged['summaries'][0]['summary'] is None:
                merged['summaries'] = temp['summaries']
            else:
                merged['summaries'] += temp['summaries']

    #Replace the old files
    if args.delete_original:
        for file in partial_files:
            if os.path.exists(file):
                os.remove(file)

    #We store the merged file
    with open(args.name, "wb") as f:
        pickle.dump(merged, f)