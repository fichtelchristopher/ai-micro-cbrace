'''
Master thesis project of Christopher Fichtel started in Oct 2022
AI in operating ortheses

File created 12 Oct 2022

Code Base parsing an aos_rulesetlibrary xml file using the xml Element Tree.

'''

import xml.etree.ElementTree as ET
import os
 

def parse_aos_rulesetlibrary(xml_file: str):
    '''
    :param xml_file: path to the aol xml file

    :return: "look up table" containing set id, rule id and ACTIVITY_COL
    '''
    assert(os.path.exists(xml_file))
    assert(xml_file.endswith("xml"))

    ruleset_dict = {}

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for ruleset in root.iter("Ruleset"):

        ruleset_name = ruleset.attrib["Name"]
        ruleset_id = ruleset.attrib["ID"]
        ruleset_id = parse_id_str(ruleset_id, ruleset_name) # now is int

        rule_dict = {}

        for rule in ruleset.iter("Rule"):

            rule_name = rule.attrib["Name"]
            rule_id = rule.attrib["ID"]
            rule_id = parse_id_str(rule_id, rule_id)

            rule_dict[rule_id] = rule_name
        
        ruleset_dict[ruleset_id] = (ruleset_name, rule_dict)

    return ruleset_dict

def parse_id_str(id_str, a): 
    '''
    ID of a ruleset in the ruleset xml is saved as '§000'. Return this value as integer.

    :param id_str: ruleset.attrib["ID"] in the xml as str
    :return: ruleset id as integer
    '''
    id_str = id_str.replace("'", "")
    id_str = id_str.replace('"', '')
    id_str = id_str.replace("§", "0")

    try:
        id_int = int(id_str)
    except:
        id_int = -1

    return id_int

    
