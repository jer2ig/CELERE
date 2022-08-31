# CELERE - Construction Evaluation for Live Reconnaisance
## An Image Analysis Algorithm for Building Integrity Assessment after Earthquakes
## Repository for Invidual Project of jh1521

Install requirements using:
```
pip install -r requirements.txt
```

Run using:
```
$ python celere.py  --weights-b damage_assessment/buildings.pt \ 
	                  --weights-d damage_assessment/damage_classify.pt \
	                  --source data/images/DamagedBuildings/
```

Or call from other python script:
```
celere.run(weights_b='damage_assessment/buildings.pt',
		   weights_d='damage_assessment/damage_classify.pt',
		   source='data/images/DamagedBuildings/')
```

The repository is a fork of ultralytics/yolov5. The principal modifications are:
- Addition of damage_assessment folder, including:
  - Pretrained model files
  - scoring_logic.py containing helper functions
  - train.py and detect.py as evidence of the custom classifier implementation
- ClassDetect.py and classify/ClassClassify.py to enable import of the models as classes
  (necessary for efficient multiple calls)
- celere.py Main script to access functionality
- annotate_image.py to annotate individual images (handy for the report)
- several minor changes to existing files for data export and to fit requirements
