import os

import matplotlib as mpl
import pandas as pd

import matplotlib.pyplot as plt

import regex as re
import numpy as np


def is_missing_columns(fp, df, necessary_columns):
    missing_columns = [
        column for column in necessary_columns if column not in df.columns
    ]

    if missing_columns:
        print(
            f"File {fp} is missing the following necessary columns: {', '.join(missing_columns)}. Skipping this file."
        )
        return True

    return False

def plot_core_v_inter(classes, years, core, inter, ylabel, fp, key):
    fig, ax = plt.subplots()

    core_count = [0] * len(years)
    inter_count = [0] * len(years)

    for k in classes.keys():
        if k in core:
            for yr in years:
                core_count[yr - 1990] += classes[k].get(f"{yr} {key}", 0)
        elif k in inter:
            for yr in years:
                inter_count[yr - 1990] += classes[k].get(f"{yr} {key}", 0)
        else:
            print("ERROR")
            exit()

    ax.plot(years, core_count, label="Core")
    ax.plot(years, inter_count, label="Interdisciplinary")

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid()

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(fp)
    plt.close(fig)


def plot_avg_core_v_inter(classes, years, core, inter):
    fig, ax = plt.subplots()

    core_cits = [0] * len(years)
    core_tot = [0] * len(years)
    inter_cits = [0] * len(years)
    inter_tot = [0] * len(years)

    for k in classes.keys():
        if k in core:
            for yr in years:
                core_cits[yr - 1990] += classes[k].get(f'{yr} cit count', 0)
                core_tot[yr - 1990] += classes[k].get(f'{yr} paper count', 0)
        elif k in inter:
            for yr in range(1990, 2021):
                inter_cits[yr - 1990] += classes[k].get(f'{yr} cit count', 0)
                inter_tot[yr - 1990] += classes[k].get(f'{yr} paper count', 0)
        else:
            print('ERROR')
            exit()

    avg_core = []
    avg_inter = []

    avg_core = [core_cits[i] / core_tot[i] if core_tot[i] > 0 else 0 for i in range(len(core_cits))]
    avg_inter = [inter_cits[i] / inter_tot[i] if inter_tot[i] > 0 else 0 for i in range(len(inter_cits))]

    ax.plot(years, avg_core, label="Core")
    ax.plot(years, avg_inter, label="Interdisciplinary")

    ax.set_xlabel("Year")
    ax.set_ylabel('Average Citations')
    ax.grid()

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('Figures/avg_citations_core_v_inter.png')
    plt.close(fig)
    

def plot_tot_papers_per_meso(classes, years):
    fig, ax = plt.subplots()
    label_colors = {'AMSC': 'deepskyblue', 'SEMI': 'blue', 'CAIM': 'blueviolet', 'DCRT': 'pink', 'WIOT': 'hotpink', 'NOCS': 'red', 'MLAI': 'forestgreen', 'SECU': 'lime', 'SSCO': 'greenyellow'}  
    keys = list(classes.keys())

    for k in keys:
        tot = []
        for yr in range(1990, 2021):
            total = classes[k].get(f"{yr} paper count", 0)
            tot.append(total)

        ax.plot(years, tot, label=k, color=label_colors[k])

    ax.set_xlabel("Year")
    ax.set_ylabel("Total Papers")
    ax.grid()

    handles, labels = plt.gca().get_legend_handles_labels()
    display_order = {'AMSC': 0, 'SEMI': 1, 'CAIM': 2, 'DCRT': 3, 'WIOT': 4, 'NOCS': 5, 'MLAI': 6, 'SECU': 7, 'SSCO': 8}  
    sorted_handles = [handles[labels.index(label)] for label in sorted(labels, key=lambda x: display_order[x])]
    sorted_labels = sorted(labels, key=lambda x: display_order[x])

    plt.legend(sorted_handles, sorted_labels, fontsize=7.5, loc='upper left')
    plt.tight_layout()
    plt.savefig("Figures/total_papers_per_meso.png", dpi=800)
    plt.close(fig)


def plot_avg_cit_per_meso(classes, years):
    fig, ax = plt.subplots()
    label_colors = {'AMSC': 'deepskyblue', 'SEMI': 'blue', 'CAIM': 'blueviolet', 'DCRT': 'pink', 'WIOT': 'hotpink', 'NOCS': 'red', 'MLAI': 'forestgreen', 'SECU': 'lime', 'SSCO': 'greenyellow'}  
    keys = list(classes.keys())

    for k in keys:
        avgs = []
        total_citations = []
        tot_paps = []
        ultimate_impact = []

        for yr in range(1990, 2021):
            cits = classes[k].get(f"{yr} cit count", 0)
            total = classes[k].get(f"{yr} paper count", 0)
            tot_paps.append(total)
            total_citations.append(cits)
            if total == 0:
                assert cits == 0
                avgs.append(0)
            else:
                avgs.append(cits / total)

        plt.plot(years, avgs, label=k, color=label_colors[k])

    ax.set_xlabel("Year")
    ax.set_ylabel("Avg # Citations")
    ax.grid()

    handles, labels = plt.gca().get_legend_handles_labels()
    display_order = {'AMSC': 0, 'SEMI': 1, 'CAIM': 2, 'DCRT': 3, 'WIOT': 4, 'NOCS': 5, 'MLAI': 6, 'SECU': 7, 'SSCO': 8}  
    sorted_handles = [handles[labels.index(label)] for label in sorted(labels, key=lambda x: display_order[x])]
    sorted_labels = sorted(labels, key=lambda x: display_order[x])

    plt.legend(sorted_handles, sorted_labels, fontsize=7.5, loc='upper left')
    plt.tight_layout()
    plt.savefig("Figures/avg_citations_per_meso.png", dpi=800)
    plt.close(fig)

def build_class_dict(fp):
    classes = {
        "NOCS": {},  
        "SEMI": {}, 
        "AMSC": {}, 
        "CAIM": {}, 
        "DCRT": {}, 
        "SECU": {}, 
        "WIOT": {}, 
        "SSCO": {}, 
        "MLAI": {}, 
    }


    necessary_columns = ["Publication Year", "Total Citations"]
    for dirpath, _, files in os.walk(fp):
        for file in files:
            if file.endswith(".xlsx") or file.endswith(".xls"):
                file_path = os.path.join(dirpath, file)
                curr_class = file_path.split("/")[-2]                           # get curr class

                try:
                    df = pd.read_excel(file_path, skiprows=range(0, 10))

                    if is_missing_columns(file_path, df, necessary_columns):
                        continue

                    for index, row in df.iterrows():                            # for each row
                        pub_year = None
                        if isinstance(row["Publication Year"], int):
                            pub_year = row["Publication Year"]
                        else:
                            print("ERROR PUB YEAR")

                        tot_cits = None
                        if isinstance(row["Total Citations"], int):
                            tot_cits = row["Total Citations"]
                        elif isinstance(row["Total Citations"], float):
                            tot_cits = int(row["Total Citations"])
                        else:
                            print("ERROR TOTAL CITS", row["Total Citations"])

                        assert classes[curr_class] is not None

                        classes[curr_class][f"{pub_year} cit count"] = (
                            classes[curr_class].get(f"{pub_year} cit count", 0) + tot_cits
                        )

                        classes[curr_class][f"{pub_year} paper count"] = (
                            classes[curr_class].get(f"{pub_year} paper count", 0) + 1
                        )

                except FileNotFoundError:
                    print(f"File {file_path} not found.")
                    continue
                except pd.errors.EmptyDataError:
                    print(f"No data in file {file_path}. Skipping this file.")
                    continue

    return classes


def get_meso_graphs():
    meso_fp = 'Analysis_Plots/Meso Classes'
    classes = build_class_dict(meso_fp)
    classes = dict(sorted(classes.items(), key=lambda x: x[0]))

    years = range(1990, 2021)

    core = [
        "SEMI",
        "NOCS",
        "DCRT",
        "CAIM",
        "WIOT",
        "AMSC",
    ]
    inter = ["SECU", "MLAI", "SSCO"]

    plot_avg_cit_per_meso(classes, years)
    plot_avg_core_v_inter(classes, years, core, inter)

    plot_tot_papers_per_meso(classes, years)
    plot_core_v_inter(classes, years, core, inter, 'Total Papers', 'Figures/total_papers_core_v_inter.png', 'paper count')
