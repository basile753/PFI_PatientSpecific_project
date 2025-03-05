"""
imports varius xmls to each specifc data

"""
# import general modules
import os
import numpy as np
from numpy import array
from numpy import linalg as linalg

from xml.etree import ElementTree as ET

import xmltodict as xtd
from meshEditting import misc

def readCylinderXML(dir):
    # empty dictioanry
    cylinderDict=dict()
    xmlTree= ET.parse(dir)
    
    # within the xml contained is cylinders with attrib
    # Name , Radius , BottomPoint, TopPoint
    root=xmlTree.getroot()
    
    for cyl in root.findall('Cylinder'):
    
        # create intermdiate dict
        temp=dict()
        
        name=cyl.find('Name').text
        rad=cyl.find('Radius').text
        bottomPoint=cyl.find('BottomPoint').text
        topPoint=cyl.find('TopPoint').text
        
        # Fill temp dict
        temp['Name']=str(name)
        temp['Rad']=rad
        temp['BottomPoint']=bottomPoint
        temp['TopPoint']=topPoint
        # fill cylinderDict
        cylinderDict[str(temp['Name'])]={'Name': temp['Name'], 'Radius': temp['Rad'], 'BottomPoint': temp['BottomPoint'],'TopPoint': temp['TopPoint']}
    
    for sph in root.findall('Sphere'):
        temp=dict()
        
        name=sph.find('Name').text
        rad=sph.find('Radius').text
        cent=sph.find('CenterPoint').text
        
        # fill temp dict
        temp['Name']=str(name)
        temp['Radius']=rad
        temp['Centre']=cent
        # add to dict
        cylinderDict[str(temp['Name'])]={'Name': temp['Name'], 'Radius' : temp['Radius'], 'Centre' : temp['Centre']}
       
    
    return cylinderDict
    
def readPointsXML(dir):
    # empty dictioanry
    XML=dict()
    xmlTree= ET.parse(dir)
    
    # within the xml contained is cylinders with attrib
    # Name , Radius , BottomPoint, TopPoint
    root=xmlTree.getroot()
    
    for pts in root.findall('Point'):
    
        # create intermdiate dict
        temp=dict()
        
        name=pts.find('Name').text
        coord=pts.find('Coordinate').text
        
        # Fill temp dict
        temp['Name']=str(name)
        temp['Coord']=misc.stringToArray(coord)[0]
        XML[str(temp['Name'])]={'Name': temp['Name'], 'Coordinate': temp['Coord']}
       
    return XML    
    
def readGenericXML(dir):

    with open(dir) as fd:
        xml = xtd.parse(fd.read())

    return xml