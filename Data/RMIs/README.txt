Last update : A. Basile, uOttawa VRS, 11_26_2024

--------data_folder----------

Here are stored the RMIs Data of the project in train/validation folders in order to train models. Both images and labelmap must be stored in the order (MPUnet documentation, Perslev):

./data_folder/
|- train/
|--- images/
|------ image1.nii.gz
|------ image5.nii.gz
|--- labels/
|------ image1.nii.gz
|------ image5.nii.gz
|- val/
|--- images/
|--- labels/

Splits can be performed for cross-validation, please to split and sort the data use the script "Data_preprocessing.py"

--------to_predict----------
	
Here are stored the RMIs data for automatic segmentation. The data must also be stored in the Following order:

./to_predict/
|-- images/
|------ image1.nii.gz
|------ image5.nii.gz
|-- labels/ #Optional
|------ image1.nii.gz
|------ image5.nii.gz