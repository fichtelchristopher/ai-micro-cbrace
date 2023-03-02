''' 
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 12 Oct 2022

Defintions for the "product" and "study" rule set libraries

The real dict will be parsed from the xml in order to dynamically be able to respond to changes.
However the 2 main rule sets are the study and product ruleset. These are dictionaries in 
this form (Oct 13 2022):

RULE_SET_STUDY = {
    0:  "1STMODE",
    1:  "TRAINING",
    2:  "FREEZE",
    3:  "MYMODE",
    4:  "BASIS",
    5:  "SIT",
    6:  "STANDBY",
    7:  "STANCEFUN",
    8:  "STANCEEXTENSION",
    9:  "STUMBLERECOVERY",
    10: "MODECHANGE",
    11: "SAFETYMODE00",
    12: "SAFETYMODE01",
    13: "SAFETYMODE02",
    14: "SAFETYMODE11",
    15: "SAFETYMODE13"
}

RULE_SET_PRODUCT = {
    0:  "1STMODE",
    1:  "TRAINING",
    2:  "FREEZE",
    3:  "MYMODE",
    4:  "ALIGN",
    5:  "SFTYPREVIEW",
    6:  "BASIS",
    7:  "SIT",
    8:  "STANDBY",
    9:  "STANCEFUN",
    10: "EXTENDEDKNEE",
    11: "STANCEEXTENSION",
    12: "MODECHANGE",
    13: "STUMBLERECOVERY",
    14: "SFTYTRANSITION",
    15: "SAFETYMODE01",
    16: "SAFETYMODE02",
    17: "SAFETYMODE11",
    18: "SAFETYMODE15"
}

# TODO double check these dicts
RULE_SET_SETID_STUDY = {
    "RI_SETID0": ["1STMODE", "TRAINING", "FREEZE", "MYMODE"], 
    "RI_SETID1": ["BASIS", "SIT", "STANDBY"], 
    "RI_SETID2": ["STANCEFUN", "STANCEEXTENSION", "STUMBLERECOVERY"],
    "RI_SETID3": ["MODECHANGE"]
}

RULE_SET_SETID_PRODUCT = {
    "RI_SETID0": ["1STMODE", "TRAINING", "FREEZE", "MYMODE", "ALIGN", "SFTYPREVIEW"], 
    "RI_SETID1": ["BASIS", "SIT", "STANDBY"], 
    "RI_SETID2": ["STANCEFUN", "EXTENDEDKNEE","STANCEEXTENSION", "STUMBLERECOVERY"],
    "RI_SETID3": ["MODECHANGE"]
}
'''


from AOS_SD_analysis.AOS_Rule_Set_Parser import parse_aos_rulesetlibrary
from Configs.namespaces import *
from AOS_SD_analysis.AOS_Rule_Set_Parser import * 

# define the path for the ruleset library xmls --> adjust these for your paths, if you dont 
# have the ruleset --> ask florian fuchs to give them to you
aos_rulesetlibrary_product =    "C:/Users/fichtel/Desktop/doku/Info/Produkt Ruleset Library/aos_rulesetlibrary.xml"
aos_rulesetlibrary_study =      "C:/Users/fichtel/Desktop/doku/Info/Studien Ruleset Library/aos_rulesetlibrary.xml"

if not os.path.exists(aos_rulesetlibrary_product):
    aos_rulesetlibrary_product =    "T:/AR/Studenten/Studenten_2022/Christopher_Fichtel/doku/info/Produkt Ruleset Library/aos_rulesetlibrary.xml"
if not os.path.exists(aos_rulesetlibrary_study):
    aos_rulesetlibrary_study =      "T:/AR/Studenten/Studenten_2022/Christopher_Fichtel/doku/info/Studien Ruleset Library/aos_rulesetlibrary.xml"

PRODUCT_RULESET = parse_aos_rulesetlibrary(aos_rulesetlibrary_product)
STUDY_RULESET = parse_aos_rulesetlibrary(aos_rulesetlibrary_study)

def get_ruleset(rule_set_library_type: str):
    '''
    Getter returning the corresponding ruleset dictionary
    integer "code" as key, string as dictionary "item"
    {
        0: ["1STMODE"   , {0: "InitRemote", 1: ...}],
        1: ["..."       , {...}],
        2: ["..."]      , {...}],
        ...
    }

    :param ruleset_type: either "product" or "study"
    :return: the ruleset dictionary 
    '''

    assert (rule_set_library_type in RULE_SET_LIBRARY_TYPES) 
    if rule_set_library_type == "product":
        ruleset = PRODUCT_RULESET 
    if rule_set_library_type == "study":
        ruleset = STUDY_RULESET

    return ruleset

def get_ruleset_key_code(rule_set_library_type:str):    
    '''
    Getter returning the corresponding ruleset dictionary
    string as key, integer code as "item"
    {
        "1STMODE: [0   , {"InitRemote": 0, "..":1 ...}],
        "...": [1,      , {"..", ...}],
        "...": [2,      , {"..", ...}],
        ...
    }
    
    :param ruleset_type: either "product" or "study"
    :return: 
    '''
    ruleset = get_ruleset(rule_set_library_type=rule_set_library_type)
    ruleset_key_code = {}

    for set_int, (set_str, rule) in ruleset.items():

        rule_key_code = {}
        for rule_int, rule_str in rule.items():
            rule_key_code[rule_str] = rule_int

        ruleset_key_code[set_str] = (set_int, rule_key_code)

    return ruleset_key_code