

ckpts/language_model/rnnlm.model.best:
	mkdir -p ckpts/language_model
	curl -L -o $@ "https://huggingface.co/xhiroga/private-models/resolve/main/raven/ckpts/language_model/rnnlm.model.best"

ckpts/vsr_prelrs3vox2avs_large_ftlrs3vox2avs_selftrain_braven.pth:
	mkdir -p ckpts
	curl -L -o $@ "https://huggingface.co/xhiroga/private-models/resolve/main/raven/ckpts/vsr_prelrs3vox2avs_large_ftlrs3vox2avs_selftrain_braven.pth"

ckpts: ckpts/language_model/rnnlm.model.best ckpts/vsr_prelrs3vox2avs_large_ftlrs3vox2avs_selftrain_braven.pth

# Variables with defaults
VIDEOS_DIR ?= data_paths/videos
LANDMARKS_DIR ?= data_paths/landmarks
TARGETS_DIR ?= data_paths/targets

normalize_videos:
	@echo "Normalizing video rotation in $(VIDEOS_DIR)"
	uv run python normalize_video_rotation.py $(VIDEOS_DIR)

extract_landmarks_fan: normalize_videos
	@echo "Processing videos from $(VIDEOS_DIR) to $(LANDMARKS_DIR)"
	@mkdir -p $(LANDMARKS_DIR)
	@find $(VIDEOS_DIR) -name "*.mp4" -o -name "*.avi" -o -name "*.mov" | while read video; do \
		basename_no_ext=$$(basename "$$video" | sed 's/\.[^.]*$$//'); \
		landmark_file="$(LANDMARKS_DIR)/$$basename_no_ext.npy"; \
		if [ ! -f "$$landmark_file" ]; then \
			echo "Processing $$video -> $$landmark_file"; \
			uv run python extract_landmarks_fan.py "$$video" "$$landmark_file"; \
		else \
			echo "Skipping $$video (landmark file already exists)"; \
		fi; \
	done

extract_mouths: extract_landmarks_fan
	@echo "Extracting mouths from $(VIDEOS_DIR) to $(TARGETS_DIR)"
	@mkdir -p $(TARGETS_DIR)
	uv run python preprocessing/extract_mouths.py \
		--src_dir $(VIDEOS_DIR) \
		--tgt_dir $(TARGETS_DIR) \
		--landmarks_dir $(LANDMARKS_DIR)

debug_visualize:
	@echo "Creating debug visualizations..."
	@mkdir -p debug_output
	@for video in $(VIDEOS_DIR)/*.mp4 $(VIDEOS_DIR)/*.mov; do \
		if [ -f "$$video" ]; then \
			basename_no_ext=$$(basename "$$video" | sed 's/\.[^.]*$$//'); \
			landmarks="$(LANDMARKS_DIR)/$$basename_no_ext.npy"; \
			cropped="$(TARGETS_DIR)/$$basename_no_ext.avi"; \
			echo "Debugging $$basename_no_ext..."; \
			uv run python debug_visualize.py \
				--video "$$video" \
				--landmarks "$$landmarks" \
				--cropped "$$cropped" \
				--output "debug_output/$$basename_no_ext"; \
		fi; \
	done


test:
	uv run test.py \
		data.modality=video \
		data/dataset=lrs3 \
		experiment_name=vsr_prelrs3vox2avs_large_ftlrs3vox2avs_selftrain_lm_braven_test \
		model/visual_backbone=resnet_transformer_large \
		model.pretrained_model_path=ckpts/vsr_prelrs3vox2avs_large_ftlrs3vox2avs_selftrain_braven.pth \
		decode.lm_weight=0.3 \
		model.pretrained_lm_path=ckpts/language_model/rnnlm.model.best
