import random

import pandas as pd
import os
import ast

import torch


def computeRatios(inFile, outFile):
    df = pd.read_csv(inFile)

    df = df[['effortsLeft', 'effortsRight', 'selected0', 'selected1']]
    # Initialize the new columns for r01, r02, and r12
    df['r01'] = 0
    df['r02'] = 0
    df['r12'] = 0

    # Loop through the dataframe and count occurrences
    for i, row in df.iterrows():
        if row['selected0'] == 0 and row['selected1'] == 1 or row['selected0'] == 1 and row['selected1'] == 0:
            df.at[i, 'r01'] += 1
        elif row['selected0'] == 0 and row['selected1'] == 2 or row['selected0'] == 2 and row['selected1'] == 0:
            df.at[i, 'r02'] += 1
        elif row['selected0'] == 1 and row['selected1'] == 2 or row['selected0'] == 2 and row['selected1'] == 1:
            df.at[i, 'r12'] += 1

    # Create a new DataFrame to store the counts
    groupedDf = df.groupby(['effortsLeft', 'effortsRight']).apply(
        lambda x: pd.Series({
            'r01': ((x['selected0'] == 0) & (x['selected1'] == 1)).sum(),
            'r02': ((x['selected0'] == 0) & (x['selected1'] == 2)).sum(),
            'r12': ((x['selected0'] == 1) & (x['selected1'] == 2)).sum()
        })
    ).reset_index()

    for i, row in groupedDf.iterrows():
        sum_row = groupedDf.at[i, 'r01'] + groupedDf.at[i, 'r02'] + groupedDf.at[i, 'r12']
        groupedDf.at[i, 'r01'] = groupedDf.at[i, 'r01'] / sum_row
        groupedDf.at[i, 'r02'] = groupedDf.at[i, 'r02'] / sum_row
        groupedDf.at[i, 'r12'] = groupedDf.at[i, 'r12'] / sum_row


    groupedDf.to_csv(outFile)

def collateRatioFiles(data_folder, outFile):
    df_total = None
    motions = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith("ratios")]
    for m in motions:
        df = pd.read_csv(m)
        df["motion"] = m.split("ratios")[1].split(".csv")[0] + "ing"
        df_total = pd.concat([df_total, df], axis=0)

    outFile = os.path.join(data_folder, outFile)
    df_total.to_csv(outFile)

def addMirroredFiles(fileName):
    df = pd.read_csv(fileName)
    # Create a copy of the original DataFrame
    df_mirror = df.copy()

    # Add the string 'Mirror' to the 'motion' column for the mirrored rows
    df_mirror['motion'] = df_mirror['motion'] + 'Mirror'

    # Concatenate the original and mirrored DataFrame
    df_combined = pd.concat([df, df_mirror], ignore_index=True)

    # Save the result to a new CSV file
    df_combined.to_csv(fileName, index=False)

def createArtificialRatios(inFile, outFile):
    df = pd.read_csv(inFile)
    # Create a copy of the original DataFrame
    df_art = df.copy()

    # Create a new DataFrame to store the counts
    for i, row in df_art.iterrows():
        effortsLeft = convert_string_to_list(df.at[i, 'effortsLeft'])
        effortsRight = convert_string_to_list(df.at[i, 'effortsRight'])
        # Highest error
        # if effortsLeft[2] == -1 and effortsRight[2] == 1:
        #     df_art.at[i, 'r01'] = 0.8
        #     df_art.at[i, 'r02'] = 0.1
        #     df_art.at[i, 'r12'] = 0.1
        # elif effortsRight[2] == -1 and effortsLeft[2] == 1:
        #     df_art.at[i, 'r01'] = 0.1
        #     df_art.at[i, 'r02'] = 0.1
        #     df_art.at[i, 'r12'] = 0.8

        # completely random
        v0 = random.random()
        v1 = (1 - v0) * random.random()
        v2 = 1 - (v1+v0)
        df_art.at[i, 'r01'] = v0
        df_art.at[i, 'r02'] = v1
        df_art.at[i, 'r12'] = v2


    df_art.to_csv(outFile)

# Convert string to list using string manipulation
def convert_string_to_list(s):
    # Remove the brackets and split by comma
    return list(map(int, s.strip('[]').split(',')))
def createEffortFiles(inFile, outFile, effortInd):
    # only read opposite efforts with effortInd: space, weight, time, flow
    df = pd.read_csv(inFile)

    df_out = pd.DataFrame(columns=df.columns)
    for i, row in df.iterrows():
        effortsLeft = convert_string_to_list(df.at[i, 'effortsLeft'])
        effortsRight = convert_string_to_list(df.at[i, 'effortsRight'])

        if  effortsLeft[effortInd] == -1 and  effortsRight[effortInd] == 1 \
            or effortsRight[effortInd] == -1 and  effortsLeft[effortInd] == 1:
            df_out = pd.concat([df_out, df.loc[[i]] ], ignore_index=True)


    df_out.to_csv(outFile)


# computeRatios('userStudyResults/validAnswersPointing.csv', 'userStudyResults/ratiosPoint.csv')
# computeRatios('userStudyResults/validAnswersWalking.csv', 'userStudyResults/ratiosWalk.csv')
# computeRatios('userStudyResults/validAnswersPicking.csv', 'userStudyResults/ratiosPick.csv')


# collateRatioFiles('userStudyResults', 'allRatios.csv')

# addMirroredFiles('userStudyResults/allRatios.csv')

# createArtificialRatios('userStudyResults/allRatios.csv', 'userStudyResults/allRatiosFake.csv')
# createEffortFiles('userStudyResults/allRatios.csv', 'userStudyResults/allRatiosSpeed.csv', 2)

#SAFE TO DELETE
# import torch.nn.functional as F
#
#
#
# df = pd.read_csv('userStudyResults/allRatios.csv')
# loss = 0
# cnt = 0
# for j in range(100):
#     for i, row in df.iterrows():
#         r01 = df.at[i, 'r01']
#         r02 = df.at[i, 'r02']
#         r12 = df.at[i, 'r12']
#
#         d_m0_m1 = random.random()
#         d_m1_m2 = (1 - d_m0_m1) * random.random()
#         d_m0_m2 = 1 - (d_m0_m1 + d_m1_m2)
#
#         loss_m0_m1 = F.mse_loss(torch.tensor([d_m0_m1]),  torch.tensor([r01]))
#         loss_m0_m2 = F.mse_loss(torch.tensor([d_m0_m2]),  torch.tensor([r02]))
#         loss_m1_m2 = F.mse_loss(torch.tensor([d_m1_m2]) , torch.tensor([r12]))
#
#         loss +=loss_m0_m1 + loss_m0_m2 + loss_m1_m2
#         cnt +=1
#
# loss/= cnt
# print(loss)
# #
# #
