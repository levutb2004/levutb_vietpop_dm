# Variables
PYTHONPATH=src
PYTHON=python
SCRIPT=vietpop.cli.main
CONFIG=prj_vn_2019/config.yaml
# MODEL=lr
#MODEL=rf
#MODEL=ensemble
#MODEL=bart
# MODEL=mlp
# MODEL=glm
MODEL=rf-pi

# Default target
train:
	@echo "Training model..."
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) -m $(SCRIPT) train -c $(CONFIG) --verbose --model-type $(MODEL)

trainmlp:
	@echo "Training model..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(SCRIPT) train -c $(CONFIG) --verbose --model-type mlp

trainlr:
	@echo "Training model..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(SCRIPT) train -c $(CONFIG) --verbose --model-type lr

run:
	@echo "Running application..."
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) -m $(SCRIPT) run -c $(CONFIG) --verbose --model-type $(MODEL)

runmlp:
	@echo "Running application..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(SCRIPT) run -c $(CONFIG) --verbose --model-type mlp

mastergrid:
	@echo "Creating mastergrid..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(SCRIPT) mastergrid -c $(CONFIG)

mergecommunes:
	@echo "Merging communes..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(SCRIPT) mergecommunes -c $(CONFIG)

spatialdiag:
	@echo "Running spatial diagnostics..."
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) -m $(SCRIPT) spatialdiag -c $(CONFIG) -m prj_vn_2019/output/$(MODEL).pkl.gz

mlflow:
	@echo "Starting MLflow UI..."
	mlflow ui --backend-store-uri prj_vn_2019/mlruns --port 5000
# 	mlflow server --backend-store-uri prj_vn_2019/mlruns --host 127.0.0.1 --port 5000 --serve-artifacts
# 	mlflow ui --backend-store-uri sqlite:///mlflow.db

featurize:
	@echo "Extracting features only..."
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m $(SCRIPT) featurize -c $(CONFIG) --verbose

predictndm:
	@echo "Predict and Dasymetric Mapping..."
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) -m $(SCRIPT) predictndm -c $(CONFIG) --verbose -m prj_vn_2019/output/$(MODEL).pkl.gz

commpopdiag:
	@echo "Running commune-level diagnostics..."
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) -m $(SCRIPT) commpopdiag -c $(CONFIG) -m prj_vn_2019/output/$(MODEL).pkl.gz --verbose

pixelpopdiag:
	@echo "Running pixel-level diagnostics..."
	set PYTHONPATH=$(PYTHONPATH) && $(PYTHON) -m $(SCRIPT) pixelpopdiag -c $(CONFIG) -m prj_vn_2019/output/$(MODEL).pkl.gz --verbose


