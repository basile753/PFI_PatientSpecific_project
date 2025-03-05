"""
imports varius xmls to each specifc data

"""
# import general modules
import os
import numpy as np
from xml.etree import ElementTree as ET

import xmltodict as xtd
 
from osimProcessing import miscTools as misc 
 
def readGenericXML(dictdir):

    with open(dictdir) as fd:
        xml = xtd.parse(fd.read() , process_namespaces=True)
        #xml = xtd.OrderedDict(fd.read())

    return xml
    
def saveDictAsXML(dict , outputFileName):

    # get a handle to the output file OI
    fo = open(outputFileName,'w')
    # unparse
    xtd.unparse(dict,fo, pretty = True)
    # clsoe and save
    fo.close()

    return
    
def readPointsXML(dir):
    # empty dictioanry
    XML=dict()
    xmlTree= ET.parse(dir)
    
    # within the xml contained is cylinders with attrib
    
    root=xmlTree.getroot()
    
    for pts in root.findall('Point'):
    
        # create intermdiate dict
        temp=dict()
        
        name=pts.find('Name').text
        coord=pts.find('Coordinate').text
        
        # Fill temp dict
        temp['Name']=str(name)
        temp['Coord']=misc.stringToArray(coord)[0]
        #XML[str(temp['Name'])]={'Name': temp['Name'], 'Coordinate': temp['Coord']}
        XML[str(temp['Name'])]= temp['Coord']
       
    return XML 