import os
import sys

sys.path.append(os.path.abspath("../.."))

import argparse

#=================================================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of preprocessing results")
    parser.add_argument("-d", action="store", type=str, default=None, help="Comma-separated list of docs. Leave blank for a predefined set of test documents. -1 for all")
    parser.add_argument("-x", nargs="?", action="store", type=str, default="default", help="Comma-separated list of experiments. Name of subdir in pickle/, images/ and /params")
    parser.add_argument("-i", action="store", type=str, default="pubmed", help="Comma-separated list of index names")
    parser.add_argument("mode", nargs="?", action="store", type=str, help="What to compare (e.g. doc means iterate over experiments, create separate tables for them, and for each experiment compare the docs)", choices=[
        "doc", #Compare documents for each separate experiment
        "exp"  #Compare experiments for each separate document
    ], default="doc")
    parser.add_argument("-db", action="store", type=str, default='mongo', help="Database to load the preprocessing results from", choices=['mongo', 'pickle'])
    parser.add_argument("--cache", action="store_true", default=False, help="Retrieve docs from cache instead of elasticsearch")
    args = parser.parse_args()

#=================================================================================================================

from collections import namedtuple
from mypackage.elastic import Session, ElasticDocument
from mypackage.helper import NpEncoder, create_table, write_to_excel_tab, DEVICE_EXCEPTION
from mypackage.sentence.metrics import chain_metrics
from mypackage.clustering.metrics import clustering_metrics, stats
from mypackage.sentence import SentenceChain
from mypackage.clustering import ChainClustering

import numpy as np

from mypackage.experiments import ExperimentManager
from mypackage.storage import PickleSession, MongoSession, DatabaseSession, ProcessedDocument

import pickle
from itertools import chain
import json

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.pretty import Pretty
from rich.padding import Padding
from rich.columns import Columns

from collections import defaultdict

import xlsxwriter

console = Console()

#=================================================================================================================

if __name__ == "__main__":

    exp_manager = ExperimentManager("../common/experiments.json")
    indexes = args.i.split(",")

    if len(indexes) > 1:
        if args.d is not None:
            raise DEVICE_EXCEPTION("THE DOCUMENTS MUST CHOOSE... TO EXIST IN ALL, IT INVITES FRACTURE.")

    #-------------------------------------------------------------------------------------------

    if args.db == "pickle":
        db = PickleSession()
    else:
        db = MongoSession()

    for index in indexes:
        console.print(f"\nRunning for index '{index}'")
        console.print(Rule())

        

        #-------------------------------------------------------------------------------------------
        os.makedirs(os.path.join(index, "stats"), exist_ok=True)
        sess = Session(index, base_path="../common", cache_dir="../cache", use="cache" if args.cache else "client")
        docs = exp_manager.get_docs(args.d, sess)
        db.base_path = os.path.join(sess.index_name, "pickles") if db.db_type == "pickle" else f"experiments_{sess.index_name}"

        if args.mode == "doc":
            workbook = xlsxwriter.Workbook("documents.xlsx")
            name_fmt = workbook.add_format({'bg_color': '#eeeeee', 'bold': True, 'border': 1})
            border_fmt = workbook.add_format({'border': 1})
            title_fmt = workbook.add_format({'border': 1, 'bg_color': '#eeeeee', 'bold': True, 'align': 'center', 'valign': 'vcenter'})

            #We iterate over all the experiments
            #For each experiment, we want to see how all documents behave
            #Only then do we move on the next experiment
            for experiment in db.available_experiments(args.x):
                db.sub_path = experiment

                chain_rows = defaultdict(list)
                cluster_rows = defaultdict(list)
                stat_rows = defaultdict(list)
                column_names = [] #The experiment names

                rich_group_items = []

                pkl = db.load(sess, docs)
                out = []

                #Iterate over the documents
                for i, p in enumerate(pkl):

                    if i == 0:
                        rich_group_items.append(Pretty(p.params))
                        rich_group_items.append(Padding(Rule(style="green"), (1,0)))

                    p: ProcessedDocument

                    #Add new column (for new experiment) to the table data
                    #-----------------------------------------------------------------------
                    column_names.append(f"{p.doc.id:04}")

                    for temp in chain_metrics(p.chains).values():
                        chain_rows[temp['name']].append(temp['value'])

                    for temp in clustering_metrics(p.clustering).values():
                        cluster_rows[temp['name']].append(temp['value'])

                    for k,v in ({'id':p.doc.id, 'index': i} | stats(p.clustering)).items():
                        stat_rows[k].append(v)

                #Write to excel file
                #-----------------------------------------------------------------------
                #Creates new tab
                worksheet = workbook.add_worksheet(experiment)

                #Populate the tab with data
                table_offset = 0
                first_col_width = 0
                write_in_rows = True

                for title, dataset in [("Chain Metrics", chain_rows), ("Clustering Metrics", cluster_rows), ("Stats", stat_rows)]:
                    if write_in_rows:
                        table_offset, first_col_width = write_to_excel_tab(worksheet, title, dataset, column_names, row_offset=table_offset, name_fmt=name_fmt, title_fmt=title_fmt, global_fmt=border_fmt, first_width=first_col_width)
                    else:
                        table_offset = write_to_excel_tab(worksheet, title, dataset, column_names, column_offset=table_offset, name_fmt=name_fmt, title_fmt=title_fmt, global_fmt=border_fmt)
                
                #Create tables
                #-----------------------------------------------------------------------
                rich_group_items.append(Padding(create_table(['Metric', *column_names], chain_rows, title="Chain Metrics"), (0,0,1,0)))
                rich_group_items.append(Padding(create_table(['Metric', *column_names], cluster_rows, title="Clustering Metrics"), (0,0,1,0)))
                rich_group_items.append(Padding(create_table(['Statistic', *column_names], stat_rows, title="Stats"), (0,0,1,0)))

                console.print(Padding(Panel(Group(*rich_group_items), title=f"Experiment: {experiment}\tIndex: {index}", border_style="green", highlight=True), (0,0,10,0)))

            workbook.close()

        #==========================================================================================================================

        elif args.mode == "exp":
            experiments = db.available_experiments(args.x)

            #if len(experiments) < 2:
                #raise Exception("I SEE, YOU INTEND FOR NO CORRELATION")
            
            if len(experiments) > 12:
                raise DEVICE_EXCEPTION("IT SPILLS BEYOND THE BRINK OF THE DEVICE")
            
            workbook = xlsxwriter.Workbook("experiments.xlsx")
            name_fmt = workbook.add_format({'bg_color': '#eeeeee', 'bold': True, 'border': 1})
            border_fmt = workbook.add_format({'border': 1})
            title_fmt = workbook.add_format({'border': 1, 'bg_color': '#eeeeee', 'bold': True, 'align': 'center', 'valign': 'vcenter'})
            
            #We iterate over all the documents
            #For each doc, we want to run and compare the selected experiments
            #Only then do we move on to the next document
            for i, doc in enumerate(docs):

                chain_rows = defaultdict(list)
                cluster_rows = defaultdict(list)
                stat_rows = defaultdict(list)
                column_names = [] #The experiment names

                #Iterate over the experiments
                for xp_num, xp_name in enumerate(experiments):
                    rich_group_items = []

                    #Open only one experiment
                    db.sub_path = xp_name
                    pkl = db.load(sess, doc)

                    #Add new column (for new experiment) to the table data
                    #-----------------------------------------------------------------------
                    column_names.append(xp_name)

                    for temp in chain_metrics(pkl.chains).values():
                        chain_rows[temp['name']].append(temp['value'])

                    for temp in clustering_metrics(pkl.clustering).values():
                        cluster_rows[temp['name']].append(temp['value'])

                    for k,v in ({'id':doc.id, 'index': i} | stats(pkl.clustering)).items():
                        stat_rows[k].append(v)

                #Write to excel file
                #-----------------------------------------------------------------------
                #Creates new tab
                worksheet = workbook.add_worksheet(f"{exp_manager.document_index(index, doc.id, i):02}. Doc {doc.id:04}")

                #Populate the tab with data
                table_offset = 0
                first_col_width = 0
                write_in_rows = False

                for title, dataset in [("Chain Metrics", chain_rows), ("Clustering Metrics", cluster_rows), ("Stats", stat_rows)]:
                    if write_in_rows:
                        table_offset, first_col_width = write_to_excel_tab(worksheet, title, dataset, column_names, row_offset=table_offset, name_fmt=name_fmt, title_fmt=title_fmt, global_fmt=border_fmt, first_width=first_col_width)
                    else:
                        table_offset = write_to_excel_tab(worksheet, title, dataset, column_names, column_offset=table_offset, name_fmt=name_fmt, title_fmt=title_fmt, global_fmt=border_fmt)
                
                #Create tables
                #-----------------------------------------------------------------------
                rich_group_items.append(Padding(create_table(['Metric', *column_names], chain_rows, title="Chain Metrics"), (0,0,1,0)))
                rich_group_items.append(Padding(create_table(['Metric', *column_names], cluster_rows, title="Clustering Metrics"), (0,0,1,0)))
                rich_group_items.append(Padding(create_table(['Statistic', *column_names], stat_rows, title="Stats"), (0,0,1,0)))

                console.print(Padding(Panel(Group(*rich_group_items), title=f"{exp_manager.document_index(index, doc.id, i):02}: Document {doc.id:04}", border_style="green", highlight=True), (0,0,10,0)))

            workbook.close()
            
        db.close()