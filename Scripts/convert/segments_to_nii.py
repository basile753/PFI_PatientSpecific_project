
import os
import slicer
from DICOMLib import DICOMUtils

# Path to the folder containing STL files and the output NIfTI file
stl_folder = r"Z:\PFI\MRI_Data\76"  # Folder containing the .stl files
output_file = r"Z:\PFI\MRI_Data\76\segmentation.nii.gz"

# Step 1: Import STL files as segments of a single segmentation
def import_stl_as_segmentation(stl_folder):
    # Create a segmentation node
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()  # To visualize the segments

    # Add each STL file as a separate segment
    for stl_file in ['Femur', 'Tibia', 'Patella', 'Femur_cartilage', 'Tibia_cartilage_medial', 'Tibia_cartilage_lateral', 'Patella_cartilage', 'Menisc_medial', 'Menisc_lateral']:
        stl_file += ".stl"
        # Load the STL file as a temporary model
        model_node = slicer.util.loadModel("Z:\PFI\MRI_Data/76/"+stl_file)
        # Add the model as a new segment in the segmentation node
        slicer.modules.segmentations.logic().ImportModelToSegmentationNode(model_node, segmentation_node)
        # Remove the temporary model
        slicer.mrmlScene.RemoveNode(model_node)
    
    return segmentation_node

# Step 2: Export the segmentation to .nii.gz
def export_segmentation_as_nifti(segmentation_node, output_file):
    # Convert the segmentation to a binary volume
    export_volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportAllSegmentsToLabelmapNode(segmentation_node, export_volume_node)
    # Save the segmentation volume in NIfTI format
    slicer.util.saveNode(export_volume_node, output_file)
    # Remove the exported node
    slicer.mrmlScene.RemoveNode(export_volume_node)

# Execute the segmentation process
segmentation_node = import_stl_as_segmentation(stl_folder)
export_segmentation_as_nifti(segmentation_node, output_file)

# Clean exit from Slicer
slicer.app.exit()
        