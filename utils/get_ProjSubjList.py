import pandas as pd
import numpy as np
import sys

# Input: VIDA_sheet.csv

def get_ProjSubjList():
    VIDA_sheet_path = str(sys.argv[1]) #  /data4/inqlee0704/VIDA_sheet_20220120.csv
    save_path = str(sys.argv[2]) # /data4/inqlee0704/ENV18PM_ProjSubjList_20220120.in
    path_prefix = str(sys.argv[3]) # /home/i243l699/temp/ImageData

    df = pd.read_csv(VIDA_sheet_path, dtype=str)
    print(f"Total rows: {len(df)}")
    # Get only VIDA done cases
    df = df[df["Status"] == "Done"].reset_index(drop=True)

    # Drop if path is empty
    df.dropna(subset=["VIDA_result_Full_Path"], inplace=True)

    # Extract columns of interests
    cols = [
        "Proj",
        "Hospital_Name",
        "StudyID",
        "CTDate",
        "InEx",
        "Thickness_mm",
        "Comment",
        "VIDA_result_Full_Path",
    ]
    df = df.loc[:, cols]
    print(f"Done: {len(df)}")

    # Get Proj
    df.loc[:, "Proj"] = df.loc[:, "Proj"].str.upper() + "PM"

    # Get Subj
    temp = df.loc[:, "StudyID"].str.split("-")
    df.loc[:, "StudyID"] = temp.str.join("")

    df["Subj"] = "PM" + df.loc[:, "Hospital_Name"] + df.loc[:, "StudyID"]

    # Rearrage columns
    cols = df.columns.tolist()
    cols = cols[:1] + cols[-1:] + cols[1:-1]
    df = df[cols]
    df = df.drop(columns=["Hospital_Name", "StudyID"])

    final_df = pd.DataFrame()
    subjs = np.unique(df.Subj)
    for subj in subjs:
        subj_df = df[df.Subj == subj].reset_index(drop=True)
        subj_df.sort_values(by=["CTDate"], inplace=True)
        subj_df_IN = subj_df[subj_df.InEx == "IN"].reset_index(drop=True)
        subj_df_EX = subj_df[subj_df.InEx == "EX"].reset_index(drop=True)
        # if len(subj_df_IN) != len(subj_df_EX):
        #     print(subj_df)
        for i in range(len(subj_df_IN)):
            subj_df_IN.InEx[i] += str(i)
        for i in range(len(subj_df_EX)):
            subj_df_EX.InEx[i] += str(i)

        final_df = pd.concat([final_df, subj_df_IN, subj_df_EX])
    final_df.sort_values(by=["Subj", "CTDate", "InEx"], inplace=True)
    final_df.drop(columns="Comment", inplace=True)

    # Rename
    final_df.rename(
        columns={"InEx": "Img", "VIDA_result_Full_Path": "ImgDir"}, inplace=True
    )
    # Replace \ with /
    final_df['ImgDir'] = final_df['ImgDir'].str.replace('\\','/')
    final_df['ImgDir'] = final_df['ImgDir'].str.replace('//','/')
    final_df['ImgDir'] = final_df['ImgDir'].str.replace('/nas1/Data2/VIDA_result',path_prefix)
    # final_df.to_csv("./ProjSubjListDateThkCmt_ENV18PM_20210922.in", sep="\t", index=False)
    final_df.to_csv(save_path, sep="\t", index=False)


if __name__ == "__main__":
    get_ProjSubjList()