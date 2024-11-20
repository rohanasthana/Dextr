cd AutoFormer/
cp ../NASLib/naslib/predictors/pruners/measures/dextr_utils/no_free_lunch_architectures lib/training_free/indicators/dextr_utils/
conda create --name autoformer python=3.7
conda activate autoformer
sed -i 's/torch/#torch/' requirements.txt
pip install torch torchvision<0.9.0 --index-url https://download.pytorch.org/whl/cu111
pip install -r requirements.txt
echo Download imagenet data and place under AutoFormer/ folder with imagenet/ name. Under imagenet you should have the train and val subfolders.
echo Be sure to have the file imagenet/validation_ground_truth.txt which contains the validation ground truth labels.
python validation_organize_in_classes.py
sed -i "s|--data-path '/home/hu15nagy/Documents/ImageNet_data/ImageNet_data| --data-path 'imagenet|" search_autoformer.sh
