import os, sys
import pandas as pd
import numpy as np

SU_NUM = "SU#"
GROUP_NUM = "Group#"

open_log_file : bool = True
outdir : str = "."

def out(message : str, type="string") :
    '''
    This method prints a message to the console and writes it to the log file.
    
    Parameters
    ----------
    message : str
        The message to print and write to the log file.
        
    type : str (optional)
        The type of message to print. Either "string" or "dataframe". Default is "string".
    
    Returns
    -------
    None
    '''
    global open_log_file, outdir
    if open_log_file:
        None if os.path.exists(outdir) else os.mkdir(outdir)
        write = 'w'
        open_log_file = False
    else:
        write = 'a'
    with open(f"{outdir}/log.txt", write) as f:
        if type == "string":
            f.write(message+"\n")
            # print(message)
        else:
            f.write(message.to_string()+"\n")
            # print(message.to_string())

def get_data(path : str, filter:bool=True) -> pd.DataFrame:
    '''
    This method reads the raw data file and returns a dataframe with the data.
    Things to check:
        - Get the most recent response from the students (submitted time)
        - check that the ratings add up to 100 exactly
    
    Parameters
    ----------
    path : str
        The path to the data file.
        
    Returns
    -------
    A dataframe with the data.
    '''
    try:
        responses = pd.read_csv(path)
    except:
        raise IOError(f"Could not find file: {path}\n Please make sure that the file path exists and correct.")

    if filter:
        # subject to how the questionaire was set
        response_headers = ['ID','Submitted on:','Username', 'Q01_Group number',\
        'Q03_Student number - self', 'Q04_Rating for yourself','Q05_Commentary for yourself',\
        'Q06_Student no. of group member #2', 'Q07_Rating for student #2','Q08_Commentary for group member #2',\
        'Q09_Student no. of group member #3', 'Q10_Rating for student #3','Q11_Commentary for group member #3',\
        'Q12_Student no. of group member #4', 'Q13_Rating for student #4','Q14_Commentary for group member #4',]
        try:
            responses = responses[response_headers]
        except:
            raise IOError(f"Error in getting headers\nPlease make sure that the header names are correct.")

        responses["Submitted on:"] = pd.to_datetime(responses["Submitted on:"], format="%d/%m/%Y %H:%M:%S")
        responses = responses.sort_values(["ID", "Submitted on:"])
        responses = responses.drop_duplicates(subset=["ID"], keep="last").reset_index(drop=True)
        responses = responses.drop(columns=["ID", "Submitted on:"])
        new_headers = ['SU#', 'Group#', 'student1','rating1','comment1','student2','rating2','comment2',\
                       'student3','rating3','comment3','student4','rating4','comment4']
        responses.rename(columns=dict(zip(responses.columns, new_headers)), inplace=True)
        for col in responses.columns:
            if col.startswith("student"):
                responses[col] = responses[col].astype(object)
        # Ensure ratings add to 100
        rating_cols = ['rating1', 'rating2', 'rating3', 'rating4']

        for idx, row in responses.iterrows():
            ratings = row[rating_cols].dropna()
            total = ratings.sum()

            if total < 100:
                # Distribute the missing points to rating2, rating3, rating4
                missing = 100 - total
                members = [col for col in rating_cols[1:] if not pd.isna(row[col])]
                distribute = missing / len(members) if members else 0

                for col in members:
                    responses.at[idx, col] += distribute

            elif total > 100:
                # Subtract the excess only from rating1
                excess = total - 100
                responses.at[idx, 'rating1'] -= excess

        responses.to_csv('files/filtered_response.csv',index=False)

    return responses


def get_groups(path : str) -> pd.DataFrame:
    '''
    This method reads the groups file and returns a dataframe with the groups and their members.
    pending : list
    
    Parameters
    ----------
    path : str
        The path to the groups file.
        
    Returns
    -------
    A dataframe with the groups and their members.
    '''
    try:
        groups = pd.read_csv(path)
    except Exception as e:
        print(e)
        raise IOError("Could not find file: "+path+"\n Please make sure that the file path exists and correct.")
    
    groups['Size'] = groups.groupby(GROUP_NUM)[SU_NUM].transform('count')
    # make sure that the ids are all strings and not given as integers
    groups["SU#"] = list(map(str,groups["SU#"].to_list()))
    groups['ERROR_FLAG'] = False
    groups['BIAS_FLAG'] = False
    return groups

    
def try_to_correct(member : str, group_members : list) -> str:
    '''
    This method attempts to correct a student's id number if it is incorrect.
    
    Parameters
    ----------
    member : str
        The incorrect id number in which to correct.     
           
    group_members : str
        A list of all the id numbers in the same group.
        
    Returns
    -------
    The corrected id number as a string.
    '''
    if member == "-" or member == "" or member == "nan":
        out("...Could not find correct id number")
        out("****************************************************")
        return None
    
    out("Attempting to correct...")
    member = int(member.ljust(8, "0"))
    correction = 0
    correction_calc = 99999999
    
    for student_id in group_members:
        id_num = int(student_id)
        calc = int(str(abs(id_num - member)).rstrip('0') or '0')
        if calc < correction_calc:
            correction = id_num
            correction_calc = calc
    
    if correction == 0:
        out("...Could not find correct id number")
        out("****************************************************")
        return None
    
    out("Changed from:\t"+str(member))
    out("Changed to: \t"+str(correction))
    out("****************************************************")
    return str(correction)
    
    
def process_students(groups : pd.DataFrame, data : pd.DataFrame, settings : dict) -> list[pd.DataFrame, pd.DataFrame]:
    '''
    This method processes every student. This is where the actual calculations are done.
    
    Parameters
    ----------
    groups : pd.DataFrame
        Dataframe that contains info on all the groups
        
    data : pd.DataFrame
        The meta data that is used to calculate the ratings

    settings : dict
        The settings used to calculate the ratings
        
    Returns
    -------
    Both the student mean and the resulting group dataframe
    '''
    global pd_response
    correct_ids = settings["correct_ids"]
    student_mean = pd.DataFrame(np.zeros((1, len(groups))), columns=groups[SU_NUM].to_list())
    ratings_given = {student: {} for student in groups[SU_NUM].astype(str)}  # student -> {peer: rating}

    for entry in data.iterrows():
        student = str(int(entry[1][SU_NUM]))
        out(f"---------- Processing student: {student} ----------")
        group = groups[groups[SU_NUM] == student]
        if not group[GROUP_NUM].to_list():
            out("Student does not belong to any groups.")
            continue
        try:
            group_size = int(group["Size"].values[0])
        except:
            group_size = 0
        group_size = group_size if group_size > 0 else 0
        group = group[GROUP_NUM].values[0]
        
        i = 0
        pending_scores = []
        pending_members = []
        duplicate = False
        while i < group_size:
            
            try:
                member = str(entry[1]["student"+str(i+1)]).split(".")[0]
                rating = float(entry[1]["rating" + str(i+1)])
                member_group = groups[groups[SU_NUM] == member][GROUP_NUM]

                if member_group.empty:
                    raise ValueError("Could not find member: "+member+" in provided groups file.")
                elif member_group.values[0] != group:
                    raise ValueError("Member: "+member+" is not in the same group as student: "+student)
                elif member in pending_members:
                    duplicate = True
                    raise ValueError("Member: "+member+" has been rated twice by student: "+student)
                

                student_mean[member] += rating / group_size
                ratings_given[student][member] = rating  # record for bias detection

                pending_scores.append(rating/ group_size)
                pending_members.append(member)
                
                out("\t"+ member+ " -> "+ str(rating))
                i += 1
                
            except Exception as e:
                print(e)
                out(f"********** ERROR IN ENTRY OF {int(entry[1][SU_NUM])} **********")
                out("Mistake detected in following input:")
                out(f"Member ID : {member}")
                out(f"Rating    : {rating}")
                if duplicate:
                    out("***** Duplicate rating *****")
                out("****************************************************")
                
                if rating == "nan" or rating == "" or rating == " " or rating == "-":
                    groups.loc[groups[SU_NUM] == student, "ERROR_FLAG"] = True
                    out("Student FLAGGED")
                    for already in range(len(pending_scores)):
                        student_mean[pending_members[already]] -= pending_scores[already]
                    i = group_size

                elif correct_ids and not duplicate:
                    correction = try_to_correct(member, groups[groups[GROUP_NUM] == group][SU_NUM].tolist())
                    
                    if correction is None:
                        groups.loc[groups[SU_NUM] == student, "ERROR_FLAG"] = True
                        out("Student FLAGGED")
                        
                        for already in range(len(pending_scores)):
                            student_mean[pending_members[already]] -= pending_scores[already]
                        i = group_size
                   
                    else:
                        entry[1]["student"+str(i+1)] = str(correction)
                        
                else:
                    groups.loc[groups[SU_NUM] == student, "ERROR_FLAG"] = True
                    out("Student FLAGGED")
                    
                    for already in range(len(pending_scores)):
                        student_mean[pending_members[already]] -= pending_scores[already]
                    i = group_size
    
    # === Bias Detection Phase === #
    if "BIAS_FLAG" not in groups.columns:
        groups["BIAS_FLAG"] = False

    for group_id in groups[GROUP_NUM].unique():
        group_members = groups[groups[GROUP_NUM] == group_id][SU_NUM].astype(str).tolist()
        if len(group_members) <= 1:
            continue

        # Collect received ratings
        member_ratings_received = {member: [] for member in group_members}
        for rater in group_members:
            for rated, score in ratings_given.get(rater, {}).items():
                if rated in member_ratings_received:
                    member_ratings_received[rated].append(score)

        avg_received = {
            member: np.mean(scores) if scores else 0
            for member, scores in member_ratings_received.items()
        }

        # Bias checking
        for rater in group_members:
            bias_found = False
            for rated in group_members:
                if rated == rater:
                    continue
                try:
                    their_rating = ratings_given[rater][rated]
                    avg_for_rated = avg_received[rated]
                    if avg_for_rated == 0:
                        continue
                    deviation = abs(their_rating - avg_for_rated) / avg_for_rated
                    if deviation > settings["bias_threshold"]:
                        groups.loc[groups[SU_NUM] == rater, "BIAS_FLAG"] = True
                        out(f"Bias flagged (Group {group_id}): {rater} -> {rated} (rating: {their_rating}, avg: {avg_for_rated:.2f})")
                        bias_found = True
                        break  # optional: break if bias found for this rater
                except KeyError:
                    continue

                    
    out("--------------------------------------------")
    return student_mean, groups


def compile_results(groups : pd.DataFrame, factors : list, create : bool):
    '''
    This method is used to compile the results and output them into one exel file for the user.
    
    Parameters
    ----------
    groups : pd.DataFrame
        The dataframe in which the results are hidden within.
        
    factors : list
        The scalling factors 
        
    create : bool   
        Will create a excel file if True
        
    Returns
    -------
    None
    '''
    global outdir
    # excel.save(groups, GROUP_NUM, outdir+"/results.xlsx")
    results = pd.DataFrame({
        "SU#": groups[SU_NUM].tolist(),
        "Group#": groups[GROUP_NUM].tolist(),    
        "Scaling factor": factors,
        "Error_Flagged": ["" if i == False else "FLAGGED" for i in groups["ERROR_FLAG"].tolist()],
        "Bias_Flagged": ["" if i == False else "FLAGGED" for i in groups["BIAS_FLAG"].tolist()],
    })
    
    errors = results[ results["Scaling factor"] == -1]
    results = results[ results["Scaling factor"] != -1]    
    results.sort_values(by=[GROUP_NUM], inplace=True, ignore_index=True)
    
    out("Final scores:")
    out("--------------------------------------------")
    out(results, type="df")
    out("--------------------------------------------")
    out("FLAGGED students:")
    out("--------------------------------------------")
    out(results[results["Error_Flagged"] == "FLAGGED"], type="df")
    out("--------------------------------------------")

    if not errors.empty:
        out("Divide by zero errors: (Did not include in final scores)")
        out(errors, type="df")
        out("--------------------------------------------")
    
    if create:
        if os.path.exists(f"{outdir}/scaling_factors.csv"):
            os.remove(f"{outdir}/scaling_factors.csv")
        df = results
        df.rename(columns={"Scaling factor": "Scaling factor 1", "Flagged": "Flagged 1"}, inplace=True)
                
        with open(f"{outdir}/scaling_factors.csv", "w") as f:
            f.write(df.to_csv(index=False))

    
def mark(args, settings: dict = {
    "capped_mark": 1.3,
    "correct_ids": True,
    "create_file": True,
    "bias_threshold": 0.3,
}):
    '''
    This script is designed to take in a specific format of a quiz pulled from SUNLearn, and automatically mark it.
    
    Parameters
    ----------
    args : str(list)
        str1 : The path to the group file
        str2 : The path to the peer review meta-data intended to be marked.
        str3 : The path to the output directory
        
    settings(optional) : dict
        capped_mark(float) : number used to set a cap to how high the final score can be
        correct_ids(boolean) : attempt to correct ids if possible, otherwise just flag students that have errors in input.
        create_file(boolean) : creates file with scores, but does not effect the output of the log file.
    Returns
    ----------
    None
    '''
    global outdir, pd_response
    outdir = "output"
    groups = get_groups(args[0])
    data = get_data(args[1])  
    # excel = ExcelOutput(groups, settings["capped_mark"])
    student_mean, groups = process_students(groups, data, settings)

    # Flag students that did not complete quiz while belonging to a group
    for i in groups[SU_NUM].tolist():
        if int(i) not in data[SU_NUM].tolist() and not groups[GROUP_NUM][groups[SU_NUM] == i].isna().values[0] :
            groups.loc[groups[SU_NUM] == i, "Flag"] = True
    
    # Assign 0 to flagged students and 100 to members in the same group
    flagged_students = groups[groups["Flag"] == True]
    for entry in flagged_students.iterrows():
        group = entry[1][GROUP_NUM]
        group_members = groups[groups[GROUP_NUM] == group][SU_NUM].tolist()
        for i in group_members:
            if i == entry[1][SU_NUM]:
                student_mean[i] += 0
            else:
                student_mean[i] += (100 / len(group_members))

    # Calculate team means
    group_names = groups[GROUP_NUM].unique().tolist()
    team_mean = pd.DataFrame(np.zeros((1, len(group_names))), columns=group_names)
    for g in group_names:
        group = groups[groups[GROUP_NUM] == g]
        if group.empty:
            continue
        size = int(group["Size"].values[0])
        team_mean[g] = student_mean[group[SU_NUM].tolist()].sum(axis=1) / size
    
    # Calculate scaling factor
    factors = []
    for entry in groups.iterrows():
        group = entry[1][GROUP_NUM]
        if team_mean[group].values[0] == 0:
            factors.append(-1)
            continue
        factor = np.minimum(student_mean[entry[1][SU_NUM]].values[0] / team_mean[group].values[0], settings["capped_mark"])
        factors.append(factor.round(2))        

    # Compiles and prints results
    compile_results(groups, factors, settings["create_file"])
    

def run_cmd():
    if len(sys.argv) > 2:
        mark(sys.argv[2:])
    else:
        paths = [
            "files/Group_allocation.csv",
            # 'files/filtered_response.csv',
            # 'files/example_response.csv',
            "files/2025_Peer_Assessment_and_Plagiarism_Declaration.csv",
            "output"   
        ]
        mark(paths) 

    print("---PEP CALCULATIONS COMPLETE. PLEASE CHECK THE OUTPUT FILE FOR RESULTS---")  