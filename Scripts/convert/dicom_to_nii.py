
import os
import slicer
from DICOMLib import DICOMUtils

# Path to the DICOM folder and the output NIfTI file
dicom_folder = "Z:\PFI\MRI_Data/76"
output_file = "Z:\PFI\MRI_Data/76/image.nii.gz"

# Step 1: Load DICOM images
def load_dicom(dicom_folder):
    # Initialize the DICOM database if it is not already open
    if not slicer.dicomDatabase.isOpen:
        slicer.dicomDatabase.initializeDatabase(os.path.join(slicer.app.temporaryPath, 'ctkDICOM.sql'))
    
    # Import the DICOM folder and load the series
    with DICOMUtils.TemporaryDICOMDatabase() as db:
        DICOMUtils.importDicom(dicom_folder, db)
        
        # Automatically load the first DICOM series found
        patient_uid = db.patients()[0]
        study_uid = db.studiesForPatient(patient_uid)[0]
        series_uid = db.seriesForStudy(study_uid)[0]
        
        loaded_node_ids = DICOMUtils.loadSeriesByUID([series_uid])
        volume_node = slicer.mrmlScene.GetNodeByID(loaded_node_ids[0]) if loaded_node_ids else None
        return volume_node

# Step 2: Export to .nii.gz
def export_volume_as_nifti(volume_node, output_file):
    if volume_node:
        slicer.util.saveNode(volume_node, output_file)

# Run the DICOM to NIfTI conversion
volume_node = load_dicom(dicom_folder)
export_volume_as_nifti(volume_node, output_file)

# Clean exit from Slicer
slicer.app.exit()
        